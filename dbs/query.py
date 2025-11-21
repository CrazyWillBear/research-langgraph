from pydantic import BaseModel, Field


class Filters(BaseModel):
    """
    Optional filters for the vector DB query.
    Supports filtering by author name and source title.
    """

    author: str | None = Field(description="Author name to filter by", default=None)
    source_title: str | None = Field(description="Source title to filter by", default=None)


class QueryAndFilters(BaseModel):
    """
    Schema for the vector DB query and filters.
    Combines a search query string with optional filter.
    """

    query: str = Field(description="Search query string", default="")
    filters: Filters | None = Field(default=None)
