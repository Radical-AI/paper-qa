from paperqa.settings import Settings, IndexSettings, AgentSettings, ParsingSettings
from paperqa.agents import build_index


def index_with_default_setting(paper_dir: str) -> Settings:
    """Get default settings with specified paper directory."""
    settings = Settings(
        temperature=0.5,
        agent=AgentSettings(
            index=IndexSettings(
                paper_directory=paper_dir,
                index_directory=f"{paper_dir}/index",
                manifest_file=f"{paper_dir}/manifest.csv",
                concurrency=12,
                batch_size=2000,
            )
        ),
        parsing=ParsingSettings(
            use_doc_details=False,
            defer_embedding=True,
            multimodal=True,
            multiprocessing_pool_enabled=True,
            multiprocessing_pool_size=12
        )
    )
    build_index(settings=settings)


if __name__ == "__main__":
    index_with_default_setting("/Users/kai/papers")
