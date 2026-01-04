"""Nightly entrypoint to refresh the knowledge base during off-peak hours.

This bypasses the daytime cooldown so scheduled jobs can always sync Drive.
"""
import os
import openai

from index_builder import sync_drive_and_rebuild_index_if_needed

openai.api_key = os.environ.get("OPENAI_API_KEY")


if __name__ == "__main__":
    did_rebuild = sync_drive_and_rebuild_index_if_needed(
        bypass_cooldown=True,
        allow_daytime_checks=True,
    )
    if did_rebuild:
        print("Nightly rebuild complete.")
    else:
        print("Nightly run skipped rebuild (no changes detected).")
