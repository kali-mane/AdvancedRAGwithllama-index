from llama_index.core.tools import FunctionTool
import os

note_file = os.path.join("data", "notes.txt")

def save_note(note):
    print(note)
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.write(note)

    return "note_saved"

notes_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="This tool can save a text based note to a file for the user"
)