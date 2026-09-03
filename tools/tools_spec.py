import arxiv
from urllib.request import urlretrieve
from typing import Dict, Literal,Optional
import os


def fetch_paper(
    query: str,
    criteria: Literal["submitteddate", "relevance", "lastUpdateddate"] = "relevance",
) -> Dict:
    """Function to retrieve the title, authors, date, summary and PDF link of an arXiv paper.

    Args:
        query: Full arXiv search query (e.g. 'ti:RAG', 'ti:"Knowledge Distillation" AND au:Hinton', 'cat:cs.AI AND ti:Agent').
        criteria: Sort criterion ('relevance', 'submitteddate', 'lastUpdateddate'). Default is 'relevance'.
    """
    mapping = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
        "submitteddate": arxiv.SortCriterion.SubmittedDate,
        "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
        "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
    }

    clean_criteria = (
        criteria.lower().replace("arxiv.sortcriterion.", "").strip()
        if isinstance(criteria, str)
        else "relevance"
    )
    sort_criterion = mapping.get(clean_criteria, arxiv.SortCriterion.Relevance)

    clean_query = query.strip() if isinstance(query, str) else "all:paper"

    client = arxiv.Client()
    search = arxiv.Search(
        query=clean_query,
        max_results=1,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = list(client.results(search))
    if not results:
        return {"error": f"No paper found for query: {clean_query}"}



    paper = results[0]
    return {
        "title": paper.title,
        "authors": [auth.name for auth in paper.authors],
        "published_date": str(paper.published),
        "summary": paper.summary,
        "pdf_url": paper.pdf_url,
        "entry_id": paper.entry_id,
    }


def download_arxiv(pdf_url: str, output_dir: str = "./downloads") -> str:
    """Function to download an arXiv paper given its PDF URL or arXiv ID.

    Args:
        pdf_url: The PDF URL or arXiv ID of the paper to download.
        output_dir: The directory where the downloaded PDF file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    url = pdf_url.strip()
    if not url.startswith("http"):
        url = f"https://arxiv.org/pdf/{url}"
    if not url.endswith(".pdf"):
        url += ".pdf"

    filename = url.split("/")[-1]
    file_path = os.path.join(output_dir, filename)

    try:
        urlretrieve(url, file_path)
        return f"Paper successfully downloaded and saved to: {file_path}"
    except Exception as e:
        return f"Error while trying to download paper from {url}: {e}"




if __name__ == "__main__":
    results = fetch_paper(query="ti:attention is all you need AND au:Geoffrey Hinton", criteria="relevance")
    print(results)
