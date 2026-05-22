"""Local HTTP API wrapping Scrapling fetchers.

POST /scrape with JSON body to extract a URL as HTML / Markdown / text.
Designed to be a drop-in self-hosted replacement for paid scraping APIs.

Environment variables
---------------------
SCRAPLING_API_KEY   If set, every request to /scrape must carry the same
                    value in the `X-API-Key` header. Strongly recommended
                    whenever the service is reachable from anything other
                    than localhost.
SCRAPLING_CORS      Comma-separated list of allowed origins for CORS.
                    Use "*" to allow any origin. Leave unset to disable
                    CORS entirely (default).
"""

import os
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scrapling.core.shell import Convertor
from scrapling.fetchers import DynamicFetcher, Fetcher, StealthyFetcher

API_KEY = os.environ.get("SCRAPLING_API_KEY", "").strip()
CORS_ORIGINS = [o.strip() for o in os.environ.get("SCRAPLING_CORS", "").split(",") if o.strip()]

app = FastAPI(title="Scrapling local API", version="1.1.0")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )

_FORMAT_MAP = {"md": "markdown", "html": "html", "text": "text"}


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
        )


class ScrapeRequest(BaseModel):
    url: str = Field(..., description="Target URL to scrape")
    mode: Literal["get", "fetch", "stealthy"] = Field(
        "get",
        description=(
            "get = plain HTTP (fastest), "
            "fetch = Playwright (JS rendering), "
            "stealthy = anti-bot (Cloudflare, fingerprint spoofing)"
        ),
    )
    format: Literal["html", "md", "text"] = "md"
    css_selector: Optional[str] = Field(
        None, description="Optional CSS selector to narrow extraction"
    )
    ai_targeted: bool = Field(
        False,
        description="Strip noise tags (script/style/hidden) — useful before feeding to an LLM",
    )
    solve_cloudflare: bool = Field(
        False, description="Only for mode='stealthy' — attempt to solve Turnstile"
    )
    proxy: Optional[str] = Field(None, description="http://user:pass@host:port")
    timeout: int = Field(30, ge=1, le=300, description="Timeout in seconds")
    wait_selector: Optional[str] = Field(
        None, description="Browser modes only — wait for this CSS selector before extracting"
    )


class ScrapeResponse(BaseModel):
    url: str
    status: int
    mode: str
    format: str
    content: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "auth_required": bool(API_KEY)}


def _fetch(req: ScrapeRequest):
    if req.mode == "get":
        kwargs = {"timeout": req.timeout}
        if req.proxy:
            kwargs["proxy"] = req.proxy
        return Fetcher.get(req.url, **kwargs)

    browser_kwargs = {"headless": True, "timeout": req.timeout * 1000}
    if req.proxy:
        browser_kwargs["proxy"] = req.proxy
    if req.wait_selector:
        browser_kwargs["wait_selector"] = req.wait_selector

    if req.mode == "fetch":
        return DynamicFetcher.fetch(req.url, **browser_kwargs)

    if req.solve_cloudflare:
        browser_kwargs["solve_cloudflare"] = True
    return StealthyFetcher.fetch(req.url, **browser_kwargs)


@app.post("/scrape", response_model=ScrapeResponse, dependencies=[Depends(require_api_key)])
def scrape(req: ScrapeRequest) -> ScrapeResponse:
    try:
        response = _fetch(req)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {exc}") from exc

    content = "".join(
        Convertor._extract_content(
            response,
            extraction_type=_FORMAT_MAP[req.format],
            css_selector=req.css_selector,
            main_content_only=req.ai_targeted,
        )
    )

    return ScrapeResponse(
        url=req.url,
        status=getattr(response, "status", 0),
        mode=req.mode,
        format=req.format,
        content=content,
    )
