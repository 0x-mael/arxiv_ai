import arxiv
from urllib.request import urlretrieve
from typing import Dict, Literal

def fetch_paper(query: str, criteria: Literal["submitteddate", "relevance", "lastUpdateddate"] ) -> Dict:
    """Function to retrieve the title, authors, date, summary and PDF link of an arXiv paper.

    Args:
        query: Search keywords or title (e.g. 'RAG', 'knowledge distillation').
        criteria: Sort criterion. Allowed values: 'relevance', 'submitted_date', 'last_updated_date'. Default is 'relevance'.
    """
    mapping = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
        "submitteddate": arxiv.SortCriterion.SubmittedDate,
        "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
        "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
    }

    # Clean and match criteria string even if the model passes variations
    clean_criteria = (
        criteria.lower().replace("arxiv.sortcriterion.", "").strip()
        if isinstance(criteria, str)
        else "relevance"
    )
    sort_criterion = mapping.get(clean_criteria, arxiv.SortCriterion.Relevance)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=1,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending, 
    )
    results = list(client.results(search))
    if not results:
        return {"error": f"No paper found for query: {query}"}

    paper = results[0]
    return {
        "title": paper.title,
        "authors": [auth.name for auth in paper.authors],
        "published_date": str(paper.published),
        "summary": paper.summary,
        "pdf_url": paper.pdf_url,
        "entry_id": paper.entry_id,
    }


def download_arxiv(paper_id_or_url: str = "") -> str:
    """Function to download an arxiv paper giving its document id or URL"""

    urlretrieve(paper_id_or_url, "paper.pdf")
    return f"Téléchargement réussi pour {paper_id_or_url}" if paper_id_or_url else "Téléchargement réussi"


if __name__ == "__main__":
    results = fetch_paper(query="RAG", criteria="relevance")
    print(results)
