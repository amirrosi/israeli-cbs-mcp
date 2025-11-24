"""
Israeli Central Bureau of Statistics (CBS) MCP Server with Cached Index

This MCP server provides fast access to Israeli statistical data by building
and caching a flat index of all available series on startup.

Installation:
    pip install mcp httpx

Usage:
    python cbs_mcp_server.py
"""

import asyncio
import httpx
import json
import os
import sys
import logging
from typing import Optional, Any, Dict, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class Config:
    """Configuration management for the server."""
    CBS_API_BASE: str = os.getenv("CBS_API_BASE", "https://apis.cbs.gov.il/series")
    CBS_INDEX_API_BASE: str = os.getenv("CBS_INDEX_API_BASE", "https://api.cbs.gov.il/index")
    CACHE_DIR: Path = Path(os.getenv("CBS_CACHE_DIR", str(Path.home() / ".cache" / "cbs_mcp")))
    CACHE_FILE: Path = CACHE_DIR / "series_index.json"
    LOG_FILE: Path = CACHE_DIR / "cbs_mcp.log"
    CACHE_DURATION: timedelta = timedelta(days=int(os.getenv("CBS_CACHE_DURATION_DAYS", "7")))
    API_TIMEOUT: float = float(os.getenv("CBS_API_TIMEOUT", "30.0"))
    MAX_RETRIES: int = int(os.getenv("CBS_MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("CBS_RETRY_DELAY", "1.0"))
    CACHE_REFRESH_INTERVAL: int = int(os.getenv("CBS_CACHE_REFRESH_INTERVAL", "86400"))  # 24 hours

# Setup logging
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)  # Use stderr for logs to avoid corrupting MCP JSON-RPC on stdout
    ]
)
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("israeli-cbs-stats")

# Global cache
path_metadata: Dict[str, Dict[str, Any]] = {}  # Map path -> {name, full_name, type}
index_ready_event = asyncio.Event()


async def fetch_cbs_api(
    endpoint: str,
    params: dict[str, Any],
    timeout: float = Config.API_TIMEOUT,
    base_url: str = Config.CBS_API_BASE
) -> dict[str, Any]:
    """Fetch data from CBS API with error handling and retries."""
    params["format"] = "json"
    params["download"] = "false"
    
    url = f"{base_url}/{endpoint}"
    logger.debug(f"Fetching: {url} with params: {params}")
    
    for attempt in range(Config.MAX_RETRIES):
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                return data
            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (client error)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"HTTP Error {e.response.status_code}: {e.response.text[:200]}")
                    return {
                        "error": f"HTTP {e.response.status_code}: {e.response.text}",
                        "url": str(e.request.url)
                    }
                logger.warning(f"Attempt {attempt + 1} failed: HTTP {e.response.status_code}")
            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return {"error": str(e)}
        
        # Wait before retrying
        if attempt < Config.MAX_RETRIES - 1:
            await asyncio.sleep(Config.RETRY_DELAY * (2 ** attempt))  # Exponential backoff

    return {"error": f"Failed after {Config.MAX_RETRIES} attempts"}


def extract_topics(result: dict) -> list:
    """Extract topics from API response, handling different response formats."""
    # Try to find topics in 'catalogs' -> 'catalog' (CBS API specific)
    if isinstance(result, dict):
        if "catalogs" in result:
            catalogs = result["catalogs"]
            if isinstance(catalogs, dict) and "catalog" in catalogs:
                return catalogs["catalog"]
            elif isinstance(catalogs, list):
                 return catalogs
        
        # Also check direct 'catalog' key just in case
        if "catalog" in result:
             return result["catalog"]

    # If result is already a list
    if isinstance(result, list):
        return result
    
    return []


async def fetch_all_topics(endpoint: str, params: dict[str, Any]) -> list[dict]:
    """Fetch all topics handling pagination."""
    all_topics = []
    page = 1
    params["PageSize"] = 500  # Increase page size to reduce requests
    
    while True:
        params["Page"] = page
        result = await fetch_cbs_api(endpoint, params)
        
        if "error" in result:
            logger.error(f"Error fetching page {page}: {result['error']}")
            break
            
        topics = extract_topics(result)
        if not topics:
            break
            
        all_topics.extend(topics)
        logger.debug(f"Fetched page {page}, got {len(topics)} topics. Total so far: {len(all_topics)}")
        
        # Check for pagination
        # Pagination info is often nested in a "paging" object
        # It can be at root, or inside "catalogs"
        paging = None
        if "paging" in result:
            paging = result["paging"]
        elif "catalogs" in result and isinstance(result["catalogs"], dict) and "paging" in result["catalogs"]:
            paging = result["catalogs"]["paging"]
            
        if not paging:
            # Fallback to top-level if not found
            paging = result
            
        last_page = paging.get("last_page")
        next_url = paging.get("next_url")
        
        if last_page and page < last_page:
            page += 1
            continue
        elif next_url:
             # If next_url is present but last_page isn't or we want to be safe
             page += 1
             continue
        else:
            break
            
    return all_topics


async def fetch_index_hierarchy(lang: str = "he") -> dict[str, dict]:
    """
    Fetch the Price Indices hierarchy using catalog/tree.
    Returns a map of path -> metadata.
    """
    logger.info("Fetching Price Indices hierarchy...")
    metadata = {}
    
    # catalog/tree returns the full hierarchy
    params = {} 
    if lang == "en":
        params["lang"] = "en"

    data = await fetch_cbs_api("catalog/tree", params, base_url=Config.CBS_INDEX_API_BASE)
    
    if "error" in data:
        logger.error(f"Failed to fetch indices: {data['error']}")
        return {}
        
    chapters = data.get("chapters", [])
    logger.info(f"Found {len(chapters)} index chapters")
    
    for chapter in chapters:
        # Level 1: Chapter
        chapter_id = chapter.get("chapterId")
        chapter_name = chapter.get("chapterName")
        
        if not chapter_id: 
            continue
            
        path_l1 = f"index:{chapter_id}"
        metadata[path_l1] = {
            "name": chapter_name,
            "full_name": chapter_name,
            "type": "Category"
        }
        
        # Level 2: Subject
        subjects = chapter.get("subject", []) or []
        for subject in subjects:
            subject_id = subject.get("subjectId")
            subject_name = subject.get("subjectName")
            
            if not subject_id:
                continue
                
            path_l2 = f"{path_l1},{subject_id}"
            full_name_l2 = f"{chapter_name} > {subject_name}"
            metadata[path_l2] = {
                "name": subject_name,
                "full_name": full_name_l2,
                "type": "Category"
            }
            
            # Level 3: Code (Series)
            codes = subject.get("code", []) or []
            for code in codes:
                code_id = code.get("codeId")
                code_name = code.get("codeName")
                
                if not code_id:
                    continue
                    
                path_l3 = f"{path_l2},{code_id}"
                full_name_l3 = f"{full_name_l2} > {code_name}"
                
                metadata[path_l3] = {
                    "name": code_name,
                    "full_name": full_name_l3,
                    "type": "Series"
                }
                
    logger.info(f"Processed {len(metadata)} index items")
    return metadata


async def build_series_index(lang: str = "he") -> dict[str, dict]:
    """
    Build a comprehensive index of all topics and series.
    Returns a map: path -> metadata (name, type, etc.)
    """
    logger.info(f"Building series index in {lang}...")
    raw_index = []
    path_names_map = {}  # Map "1,2,3" -> "Subject > Sub > Item"
    
    # Step 1: Fetch Level 1 (Main Subjects)
    # Use fetch_all_topics to handle potential pagination even at level 1
    level1_topics = await fetch_all_topics("catalog/level", {"id": 1, "lang": lang})
    
    logger.info(f"Found {len(level1_topics)} level 1 topics")
    
    # Step 2: Process each subject
    for topic1 in level1_topics:
        # Extract ID from path
        path_list = topic1.get("path", [])
        if not path_list:
            continue
            
        subject_id = path_list[-1]
        name1 = topic1.get("Name") or topic1.get("name", "Unknown")
        
        # Add to maps and index
        path_str = str(subject_id)
        path_names_map[path_str] = name1
        
        raw_index.append({
            "series_id": path_str,
            "series_name": name1,
            "full_name": name1
        })
        
        logger.info(f"Processing Subject: {name1} (ID: {subject_id})")
        
        # Iterate levels 2 to 5 for this subject
        for level in range(2, 6):
            params = {"id": level, "subject": subject_id, "lang": lang}
            
            # Use fetch_all_topics to handle pagination
            topics = await fetch_all_topics("catalog/level", params)
            
            if not topics:
                continue
                
            logger.debug(f"Subject {subject_id} Level {level}: {len(topics)} items")
            
            for topic in topics:
                path_list = topic.get("path", [])
                if not path_list:
                    continue
                
                # Filter out paths containing None
                if any(p is None for p in path_list):
                    continue

                current_path_str = ",".join(map(str, path_list))
                name = topic.get("Name") or topic.get("name", "Unknown")
                
                # Construct full name using parent's name
                parent_path_str = ",".join(map(str, path_list[:-1]))
                parent_name = path_names_map.get(parent_path_str, "")
                                
                if parent_name:
                    full_name = f"{parent_name} > {name}"
                else:
                    full_name = name
                
                path_names_map[current_path_str] = full_name
                
                raw_index.append({
                    "series_id": current_path_str,
                    "series_name": name,
                    "full_name": full_name
                })
    
    logger.info(f"Index built: {len(raw_index)} items found")
    
    # Step 3: Identify Categories vs Series
    # Optimized to O(N)
    
    # First pass: Identify all paths that are parents of other paths
    parents_with_children = set()
    for item in raw_index:
        current_path = item["series_id"]
        # If this path has a parent, the parent is a prefix
        if "," in current_path:
            parent_path = current_path.rsplit(",", 1)[0]
            parents_with_children.add(parent_path)

    # Second pass: Build final metadata map
    final_metadata = {}
    for item in raw_index:
        path = item["series_id"]
        is_category = path in parents_with_children
        
        final_metadata[path] = {
            "name": item["series_name"],
            "full_name": item["full_name"],
            "type": "Category" if is_category else "Series"
        }
            
    # --- Part 2: Price Indices (New Logic) ---
    indices_metadata = await fetch_index_hierarchy(lang)
    final_metadata.update(indices_metadata)

    logger.info(f"Total Metadata built: {len(final_metadata)} items")
    return final_metadata


def save_cache(metadata: dict, lang: str):
    """Save the series index to cache."""
    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        "metadata": metadata,
        "built_at": datetime.now().isoformat(),
        "language": lang,
        "count": len(metadata)
    }
    
    with open(Config.CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Cache saved to {Config.CACHE_FILE} with {len(metadata)} items")


def load_cache() -> dict:
    """Load the series index from cache if available and recent."""
    if not Config.CACHE_FILE.exists():
        logger.info("No cache file found")
        return {}
    
    try:
        with open(Config.CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        built_at = datetime.fromisoformat(cache_data["built_at"])
        
        # Check if cache is still valid
        if datetime.now() - built_at > Config.CACHE_DURATION:
            logger.info("Cache expired")
            return {}
        
        logger.info(f"Loaded cache: {cache_data['count']} items from {cache_data['built_at']}")
        return cache_data.get("metadata", {})
    
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return {}


async def ensure_index_ready(lang: str = "he"):
    """Ensure the series index is built and loaded."""
    global path_metadata
    
    if path_metadata:
        index_ready_event.set()
        return  # Already loaded
    
    # Try to load from cache
    path_metadata = load_cache()
    
    if path_metadata:
        index_ready_event.set()
        return

    # Build new index
    logger.info("Starting background index build...")
    try:
        path_metadata = await build_series_index(lang)
        save_cache(path_metadata, lang)
    except Exception as e:
        logger.error(f"Background index build failed: {e}")
    finally:
        index_ready_event.set()


async def refresh_cache_loop():
    """Background task to refresh the cache periodically."""
    while True:
        try:
            await asyncio.sleep(Config.CACHE_REFRESH_INTERVAL)
            logger.info("Starting scheduled cache refresh...")
            
            # Rebuild index
            new_metadata = await build_series_index()
            
            if new_metadata:
                global path_metadata
                path_metadata = new_metadata
                save_cache(path_metadata, "he")
                logger.info("Cache refresh completed successfully")
            else:
                logger.warning("Cache refresh returned empty metadata, keeping old cache")
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cache refresh loop: {e}")
            await asyncio.sleep(300)  # Retry after 5 minutes on error


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="list_main_topics",
            description=(
                "Get the list of main statistical topics (Level 1) AND Price Indices chapters. "
                "Use this to start exploring the data hierarchy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lang": {
                        "type": "string",
                        "enum": ["he", "en"],
                        "default": "he"
                    }
                }
            }
        ),
        Tool(
            name="list_subtopics",
            description=(
                "Get the subtopics for a specific topic. "
                "Provide the 'parent_id' (path) from a previous list_topics call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_id": {
                        "type": "string",
                        "description": "The ID (path) of the parent topic (e.g., '8' or 'index:a')"
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["he", "en"],
                        "default": "he"
                    }
                },
                "required": ["parent_id"]
            }
        ),
        Tool(
            name="get_series_data",
            description=(
                "Get data for a specific series. "
                "You can filter by years (start_year/end_year) OR get the last N items."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "series_id": {
                        "type": "string",
                        "description": "The ID (path) of the series (e.g., '8,1,2,1' or 'index:a,37,120010')"
                    },
                    "start_year": {
                        "type": "integer",
                        "description": "Start year (e.g., 2010)"
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "End year (e.g., 2020)"
                    },
                    "last": {
                        "type": "integer",
                        "description": "Get the last N items (e.g., 12 for last year's monthly data)"
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["he", "en"],
                        "default": "he"
                    }
                },
                "required": ["series_id"]
            }
        ),
        Tool(
            name="rebuild_index",
            description="Rebuild the internal search index. Use only if data seems stale.",
            inputSchema={
                "type": "object",
                "properties": {
                    "lang": {
                        "type": "string",
                        "enum": ["he", "en"],
                        "default": "he"
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    global path_metadata
    lang = arguments.get("lang", "he")
    
    # Wait for index if not ready (needed for all tools now)
    if not index_ready_event.is_set():
        if not Config.CACHE_FILE.exists():
             return [TextContent(type="text", text="Index is building. Please wait.")]
        try:
            await asyncio.wait_for(index_ready_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
             return [TextContent(type="text", text="Index is loading. Please wait.")]
    
    if name == "list_main_topics":
        # Filter for Level 1 topics (no commas in path)
        # For indices, level 1 is "index:X" (no commas)
        topics = []
        for path, meta in path_metadata.items():
            if "," not in path:
                topics.append((path, meta))
        
        # Sort: Put numeric IDs first, then index IDs
        def sort_key(item):
            path = item[0]
            if path.startswith("index:"):
                return (1, path)
            try:
                return (0, int(path))
            except:
                return (0, path)
                
        topics.sort(key=sort_key)
        
        if not topics:
            return [TextContent(type="text", text="No main topics found.")]
            
        output = "Main Statistical Topics & Indices:\n\n"
        for path, meta in topics:
            type_label = "Category" if meta["type"] == "Category" else "Series"
            output += f"- **{meta['name']}** (ID: `{path}`) [{type_label}]\n"
            
        return [TextContent(type="text", text=output)]
    
    elif name == "list_subtopics":
        parent_id = arguments["parent_id"]
        
        # Find direct children
        children = []
        parent_prefix = parent_id + ","
        
        for path, meta in path_metadata.items():
            # Check if it starts with parent_prefix and has no more commas after that
            if path.startswith(parent_prefix):
                suffix = path[len(parent_prefix):]
                if "," not in suffix:
                    children.append((path, meta))
        
        if not children:
             return [TextContent(type="text", text=f"No subtopics found for ID {parent_id}.")]
        
        # Sort by ID suffix
        def get_suffix(p):
            try:
                return int(p.split(",")[-1])
            except:
                return 0
        children.sort(key=lambda x: get_suffix(x[0]))
             
        output = f"Subtopics for ID {parent_id}:\n\n"
        for path, meta in children:
            type_label = "Category" if meta["type"] == "Category" else "Series"
            output += f"- **{meta['name']}** (ID: `{path}`) [{type_label}]\n"
            
        return [TextContent(type="text", text=output)]
    
    elif name == "get_series_data":
        series_id = arguments["series_id"]
        
        # Check if it's a series
        meta = path_metadata.get(series_id)
        if meta and meta["type"] == "Category":
             return [TextContent(type="text", text=f"ID {series_id} is a Category, not a Series. Please use list_subtopics to find a Series.")]
        
        # Validate inputs
        if "start_year" in arguments and "end_year" in arguments:
            if arguments["start_year"] > arguments["end_year"]:
                return [TextContent(type="text", text="Error: start_year cannot be greater than end_year.")]
        
        # Determine if it's a regular series or an index
        is_index = series_id.startswith("index:")
        
        params = {
            "lang": lang,
            "addNull": "true"
        }
        
        if is_index:
            # For indices, ID is the last part of the path (the code)
            # e.g., index:a,37,120010 -> 120010
            params["id"] = series_id.split(",")[-1]
            endpoint = "data/price"
            base_url = Config.CBS_INDEX_API_BASE
        else:
            # For regular series, ID is the full path
            params["id"] = series_id
            endpoint = "data/path"
            base_url = Config.CBS_API_BASE
        
        # Handle Year Range
        if "start_year" in arguments:
            params["startPeriod"] = f"01-{arguments['start_year']}"
        if "end_year" in arguments:
            params["endPeriod"] = f"12-{arguments['end_year']}"
            
        # Handle Last N items
        if "last" in arguments:
            params["last"] = arguments["last"]
            params["addNull"] = "false" 
            
        result = await fetch_cbs_api(endpoint, params, base_url=base_url)
        
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]


    elif name == "rebuild_index":
        path_metadata = {}
        index_ready_event.clear()
        asyncio.create_task(ensure_index_ready(lang))
        return [TextContent(type="text", text="Index rebuild started.")]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    # Build/load index on startup
    logger.info("=" * 60)
    logger.info("Initializing CBS MCP Server...")
    logger.info(f"Cache directory: {Config.CACHE_DIR}")
    logger.info(f"Log file: {Config.LOG_FILE}")
    logger.info("=" * 60)
    
    # Start index build in background
    asyncio.create_task(ensure_index_ready("he"))
    
    # Start background refresh loop
    asyncio.create_task(refresh_cache_loop())
    
    logger.info("Server ready! (Index building in background)")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())