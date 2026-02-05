"""
RoadGraph — Big city portion (Austin) using a BOUNDING BOX
Compatible with newer OSMnx APIs (bbox tuple + distance.add_edge_lengths)

Outputs:
- road_graph.png
- shortest_route.png
- road_map.html (interactive: roads + ALL nodes + route)
- route_with_buildings.png (optional)

If the HTML is slow for huge graphs, reduce BBOX_DIST_METERS or NODE_RADIUS.
"""

from __future__ import annotations

import sys
from typing import Tuple, Dict, Any, List

import networkx as nx
import matplotlib.pyplot as plt

import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
import folium


# ----------------------------
# CONFIG
# ----------------------------
PLACE_NAME = "Austin, Texas, USA"

# Half-size of the bbox in meters:
# 8000 -> ~16km x 16km; try 12000 or 20000 for bigger (can get heavy).
BBOX_DIST_METERS = 8000

NETWORK_TYPE = "drive"  # "drive", "walk", "bike", etc.

CENTER_OVERRIDE: Tuple[float, float] | None = None  # set None to auto-center

# Source/destination within Austin (lat, lon)
SOURCE_POINT = (30.2747, -97.7404)  # near Capitol
DEST_POINT = (30.2643, -97.7473)    # near Lady Bird Lake

# Buildings overlay (optional)
BUILDING_DIST_METERS = 400

# Node visualization on Folium map
NODE_RADIUS = 1.8
NODE_OPACITY = 0.85


# ----------------------------
# Compatibility helpers
# ----------------------------
def geocode_compat(query: str) -> Tuple[float, float]:
    """
    OSMnx has moved some geocoding functions across versions.
    Try the common locations.
    """
    if hasattr(ox, "geocode"):
        return ox.geocode(query)

    # newer versions may have geocoder module
    if hasattr(ox, "geocoder") and hasattr(ox.geocoder, "geocode"):
        return ox.geocoder.geocode(query)

    raise RuntimeError("Could not find a geocode function in your OSMnx install.")


def add_edge_lengths_compat(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    In OSMnx 2.x, add_edge_lengths is in ox.distance.add_edge_lengths. :contentReference[oaicite:1]{index=1}
    In older versions, it may exist at ox.add_edge_lengths.
    """
    if hasattr(ox, "distance") and hasattr(ox.distance, "add_edge_lengths"):
        return ox.distance.add_edge_lengths(G)

    if hasattr(ox, "add_edge_lengths"):
        return ox.add_edge_lengths(G)

    # If neither exists, we'll assume lengths already exist.
    return G


def basic_stats_compat(G: nx.MultiDiGraph, area: float | None = None) -> Dict[str, Any]:
    if hasattr(ox, "basic_stats"):
        return ox.basic_stats(G, area=area) if area is not None else ox.basic_stats(G)
    if hasattr(ox, "stats") and hasattr(ox.stats, "basic_stats"):
        return ox.stats.basic_stats(G, area=area) if area is not None else ox.stats.basic_stats(G)
    return {}


def buildings_from_address_compat(address: str, dist: int) -> gpd.GeoDataFrame:
    tags = {"building": True}

    if hasattr(ox, "features_from_address"):
        return ox.features_from_address(address, tags=tags, dist=dist)

    if hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_address"):
        return ox.geometries.geometries_from_address(address, tags=tags, dist=dist)

    if hasattr(ox, "geometries_from_address"):
        return ox.geometries_from_address(address, tags=tags, dist=dist)

    raise RuntimeError("Could not find a buildings-from-address function in your OSMnx version.")


# ----------------------------
# Core pipeline (BBOX download)
# ----------------------------
def build_graph_bbox(place: str, bbox_dist_m: int, network_type: str) -> tuple[nx.MultiDiGraph, Tuple[float, float]]:
    """
    Geocode -> bbox_from_point -> graph_from_bbox(bbox_tuple) -> add edge lengths -> project graph
    """
    center_latlon = geocode_compat(place)
    lat, lon = center_latlon

    north, south, east, west = ox.utils_geo.bbox_from_point((lat, lon), dist=bbox_dist_m)
    bbox = (north, south, east, west)  # newer API expects bbox tuple

    G = ox.graph_from_bbox(
        bbox,
        network_type=network_type,
        simplify=True,
    )

    # Ensure 'length' attribute exists
    G = add_edge_lengths_compat(G)

    # Project to meters for routing/stat area correctness
    G = ox.project_graph(G)
    return G, center_latlon


def plot_road_graph(G: nx.MultiDiGraph, out_png: str) -> None:
    fig, ax = ox.plot_graph(G, figsize=(10, 10), node_size=5, edge_linewidth=0.8, show=False, close=False)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def graph_to_gdfs(G: nx.MultiDiGraph) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return ox.graph_to_gdfs(G)


def print_some_info(G: nx.MultiDiGraph, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> None:
    print("\n--- Graph info ---")
    print(f"Nodes: {len(G):,}")
    print(f"Edges: {len(edges):,}")

    if "highway" in edges.columns:
        print("\n--- highway value counts (top 20) ---")
        print(edges["highway"].value_counts().head(20))


def compute_stats(G: nx.MultiDiGraph, edges: gpd.GeoDataFrame) -> None:
    print("\n--- Stats ---")
    try:
        hull = edges.geometry.union_all().convex_hull
        area = hull.area  # projected => square meters
        stats = basic_stats_compat(G, area=area)
        for k in ["n", "m", "k_avg", "edge_length_total", "street_length_total"]:
            if k in stats:
                print(f"{k}: {stats[k]}")
    except Exception as e:
        print("(Stats skipped)", repr(e))


def nearest_nodes_and_shortest_route(
    G: nx.MultiDiGraph,
    source_latlon: Tuple[float, float],
    dest_latlon: Tuple[float, float],
) -> Tuple[int, int, List[int]]:
    X = [source_latlon[1], dest_latlon[1]]  # lon
    Y = [source_latlon[0], dest_latlon[0]]  # lat

    u, v = ox.distance.nearest_nodes(G, X, Y)
    route = nx.shortest_path(G, source=u, target=v, weight="length")
    return u, v, route


def plot_route(G: nx.MultiDiGraph, route: List[int], out_png: str) -> None:
    fig, ax = ox.plot_graph_route(G, route, figsize=(12, 12), route_linewidth=4, node_size=0, show=False, close=False)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Folium drawing (roads + ALL nodes + route)
# ----------------------------
def _add_graph_edges_to_folium(edges_ll: gpd.GeoDataFrame, fmap: folium.Map, weight: int = 2, opacity: float = 0.5):
    for geom in edges_ll.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coords = [(lat, lon) for lon, lat in geom.coords]
            folium.PolyLine(coords, weight=weight, opacity=opacity).add_to(fmap)
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = [(lat, lon) for lon, lat in line.coords]
                folium.PolyLine(coords, weight=weight, opacity=opacity).add_to(fmap)


def _add_nodes_to_folium(nodes_ll: gpd.GeoDataFrame, fmap: folium.Map, radius: float, opacity: float):
    for _, row in nodes_ll.iterrows():
        folium.CircleMarker(
            location=(row.geometry.y, row.geometry.x),  # (lat, lon)
            radius=radius,
            color="blue",
            fill=True,
            fill_opacity=opacity,
            weight=0,
        ).add_to(fmap)


def _add_route_to_folium(nodes_ll: gpd.GeoDataFrame, route: List[int], fmap: folium.Map, weight: int = 6, opacity: float = 0.9):
    route_nodes = nodes_ll.loc[route]
    coords = [(pt.y, pt.x) for pt in route_nodes.geometry.values]
    folium.PolyLine(coords, weight=weight, opacity=opacity).add_to(fmap)


def save_interactive_map(
    G_projected: nx.MultiDiGraph,
    center_latlon: Tuple[float, float],
    source_latlon: Tuple[float, float],
    dest_latlon: Tuple[float, float],
    route: List[int],
    out_html: str,
) -> None:
    # Convert projected graph back to lat/lon for web mapping
    G_ll = ox.project_graph(G_projected, to_latlong=True)
    nodes_ll, edges_ll = ox.graph_to_gdfs(G_ll, nodes=True, edges=True, fill_edge_geometry=True)

    m = folium.Map(location=center_latlon, zoom_start=12, tiles="cartodbpositron")

    folium.Marker(location=source_latlon, tooltip="Source").add_to(m)
    folium.Marker(location=dest_latlon, tooltip="Destination").add_to(m)

    _add_graph_edges_to_folium(edges_ll, m, weight=2, opacity=0.45)
    _add_nodes_to_folium(nodes_ll, m, radius=NODE_RADIUS, opacity=NODE_OPACITY)
    _add_route_to_folium(nodes_ll, route, m, weight=6, opacity=0.9)

    m.save(out_html)


# ----------------------------
# Optional: route + buildings (static PNG)
# ----------------------------
def plot_route_with_buildings(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    route: List[int],
    address: str,
    out_png: str,
    building_dist: int,
) -> None:
    buildings = buildings_from_address_compat(address, dist=building_dist)
    buildings = buildings.to_crs(edges.crs)

    route_nodes = nodes.loc[route]
    route_line = LineString(list(route_nodes.geometry.values))
    route_geom = gpd.GeoDataFrame([[route_line]], geometry="geometry", crs=edges.crs)

    ax = edges.plot(linewidth=0.6, color="gray", figsize=(15, 15))
    ax = buildings.plot(ax=ax, facecolor="khaki", alpha=0.7)
    ax = route_geom.plot(ax=ax, linewidth=2.2, linestyle="--", color="red")

    fig = ax.get_figure()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("PYTHON EXECUTABLE:", sys.executable)
    print("PYTHON VERSION:", sys.version)

    print("Downloading road network (bbox)...")
    G, center_latlon = build_graph_bbox(PLACE_NAME, BBOX_DIST_METERS, NETWORK_TYPE)

    center_to_use = CENTER_OVERRIDE if CENTER_OVERRIDE is not None else center_latlon

    plot_road_graph(G, "road_graph.png")
    print("Saved: road_graph.png")

    nodes, edges = graph_to_gdfs(G)

    print_some_info(G, nodes, edges)
    compute_stats(G, edges)

    print("\nComputing shortest route...")
    u, v, route = nearest_nodes_and_shortest_route(G, SOURCE_POINT, DEST_POINT)
    print(f"Nearest source node: {u}")
    print(f"Nearest dest node:   {v}")
    print(f"Route nodes count:   {len(route)}")

    plot_route(G, route, "shortest_route.png")
    print("Saved: shortest_route.png")

    save_interactive_map(G, center_to_use, SOURCE_POINT, DEST_POINT, route, "road_map.html")
    print("Saved: road_map.html")

    try:
        plot_route_with_buildings(nodes, edges, route, PLACE_NAME, "route_with_buildings.png", BUILDING_DIST_METERS)
        print("Saved: route_with_buildings.png")
    except Exception as e:
        print("\n(Warning) Could not plot buildings overlay:", repr(e))
        print("You still have road_graph.png, shortest_route.png, and road_map.html")

    print("\nDone ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
