from agents import function_tool
from agents_lib.models import tavily_client


@function_tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web and return a list of result snippets with URLs.

    Args:
        query: The search query. Keep it short and specific (3-8 words).
        max_results: How many results to return. Default 5, max 10.
    """
    try:
        response = await tavily_client.search(
            query=query, max_results=max_results, search_depth="basic"
        )
    except Exception as e:
        return f"Search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, start=1):
        formatted.append(
            f"[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r['content'][:300]}"
        )
    return "\n\n".join(formatted)
