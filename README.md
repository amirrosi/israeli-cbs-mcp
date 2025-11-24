# Israeli CBS MCP Server

An [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that provides access to statistical data from the **Israeli Central Bureau of Statistics (CBS)**.

It supports both **Statistical Series** (e.g., Unemployment, GDP) and **Price Indices** (e.g., CPI, Construction Inputs).

## Data Sources & API Documentation

This server retrieves data directly from the official CBS APIs:

*   **Statistical Series API**: [Documentation](https://www.cbs.gov.il/he/cbsNewBrand/Pages/%D7%A1%D7%93%D7%A8%D7%95%D7%AA-%D7%A2%D7%99%D7%AA%D7%99%D7%95%D7%AA-%D7%91%D7%90%D7%9E%D7%A6%D7%A2%D7%95%D7%AA-API.aspx)
*   **Price Indices API**: [Documentation](https://www.cbs.gov.il/he/cbsNewBrand/Pages/%D7%9E%D7%93%D7%93%D7%99-%D7%9E%D7%97%D7%99%D7%A8%D7%99%D7%9D-%D7%91%D7%90%D7%9E%D7%A6%D7%A2%D7%95%D7%AA-API.aspx)

## Features

-   **Hierarchical Exploration**: Navigate topics and subtopics (`list_main_topics`, `list_subtopics`).
-   **Unified Search**: Access both Statistical Series and Price Indices.
-   **Data Retrieval**: Fetch time-series data with year filtering (`get_series_data`).
-   **Smart Caching**: Builds a local index for fast lookups and offline navigation.
-   **Auto-Refresh**: Keeps the cache up-to-date with a background task.

## Indexing Behavior

Upon startup, the server traverses the entire CBS catalog to build a comprehensive local index.
- **Duration**: This process takes approximately **5 minutes**.
- **Cache Size**: The resulting index file is about **20MB**.
- **Availability**: The server is available immediately, but search/navigation tools will report "Index is building" until completion.

## Installation

### Using `uv` (Recommended)

```bash
uvx israeli-cbs-mcp
```

### From Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/amirrosi/israeli-cbs-mcp.git
    cd israeli-cbs-mcp
    ```
2.  Install dependencies:
    ```bash
    pip install .
    ```

## Configuration

You can configure the server using environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CBS_CACHE_DIR` | Directory to store the index cache | `~/.cache/cbs_mcp` |
| `CBS_CACHE_DURATION_DAYS` | How long to keep the cache before full rebuild | `7` |
| `CBS_CACHE_REFRESH_INTERVAL` | Background refresh interval (seconds) | `86400` (24h) |
| `CBS_API_TIMEOUT` | API timeout in seconds | `30.0` |
| `CBS_MAX_RETRIES` | Number of API retries | `3` |

## Usage with Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cbs": {
      "command": "uvx",
      "args": ["israeli-cbs-mcp"]
    }
  }
}
```

## Tools

-   `list_main_topics(lang="he")`: List top-level categories.
-   `list_subtopics(parent_id, lang="he")`: List children of a category.
-   `get_series_data(series_id, start_year, end_year, last)`: Get data points.
-   `rebuild_index(lang="he")`: Force a manual index rebuild.

## License

MIT
