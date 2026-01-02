import sqlite3
import argparse
import os
from contextlib import contextmanager
from langgraph.checkpoint.sqlite import SqliteSaver

@contextmanager
def get_saver(db_path: str):
    """Context manager to setup SqliteSaver with a connection."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        # Initialize SqliteSaver with the connection
        # It handles serialization/deserialization automatically
        saver = SqliteSaver(conn)
        yield saver
    finally:
        conn.close()

def inspect_checkpoints(db_path: str, limit: int = 5, thread_id: str = None):
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    print(f"Connecting to database: {db_path}")
    
    try:
        with get_saver(db_path) as saver:
            # Construct config for filtering if thread_id is provided
            config = None
            if thread_id:
                config = {"configurable": {"thread_id": thread_id}}
            
            # Use the saver's list method to get history
            # This returns an iterator of CheckpointTuple objects
            # CheckpointTuple contains: config, checkpoint, metadata, parent_config, pending_writes
            print(f"Fetching checkpoints (limit={limit}, thread_id={thread_id})...")
            checkpoint_tuples = list(saver.list(config, limit=limit))

            if not checkpoint_tuples:
                print("No checkpoints found.")
                return

            print(f"Found {len(checkpoint_tuples)} checkpoint(s). Showing oldest to newest:\n")
            
            # Reverse the list to show in chronological order
            checkpoint_tuples.reverse()
            
            for i, cp_tuple in enumerate(checkpoint_tuples):
                print(f"--- Record {i+1} ---")
                config = cp_tuple.config
                checkpoint = cp_tuple.checkpoint
                metadata = cp_tuple.metadata
                
                t_id = config.get("configurable", {}).get("thread_id", "N/A")
                c_id = checkpoint.get("id", "N/A")
                ts = checkpoint.get("ts", "N/A")
                
                print(f"Thread ID:     {t_id}")
                print(f"Checkpoint ID: {c_id}")
                print(f"Timestamp:     {ts}")
                
                print("\n[Checkpoint Data (Channel Values)]")
                # Automatically decoded channel values
                print(checkpoint.get("channel_values", {}))
                
                print("\n[Metadata]")
                print(metadata)
                
                if cp_tuple.pending_writes:
                    print("\n[Pending Writes]")
                    print(cp_tuple.pending_writes)
                    
                print("-" * 50 + "\n")
                
    except Exception as e:
        print(f"Error inspecting checkpoints: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LangGraph SQLite Checkpoints using LangGraph API")
    parser.add_argument("db_path", nargs="?", default="../checkpoints/checkpoints.sqlite", help="Path to the sqlite database file")
    parser.add_argument("--limit", type=int, default=20, help="Number of checkpoints to show")
    parser.add_argument("--thread", type=str, help="Filter by thread_id")
    
    args = parser.parse_args()
    inspect_checkpoints(args.db_path, args.limit, args.thread)

# Example usage:
# uv run tutorials/utils/inspect_checkpoint.py tutorials/checkpoints/checkpoints.sqlite --thread user_neo --limit 20