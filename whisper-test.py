import os
import gc
import torch
import whisper

def retranscript(audio_path, model_name="base"):
    """
    Retranscribes the given audio file using the specified model.

    Args:
        audio_path (str): The path to the audio file.
        model_name (str, optional): The name of the model to use for transcription. Defaults to "base".

    Returns:
        tuple: A tuple containing the transcribed text and a boolean indicating the success of the transcription.
    """
    global success
    success = False

    # With the Whisper library
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name).to(device)
        options = whisper.DecodingOptions(fp16=False)
        result = model.transcribe(
            audio_path,
            temperature=0.0,
            word_timestamps=False,
            decode_options=options
        )

        del model
        gc.collect()

        result = result["text"]

        success = True

    except Exception as e:
        result = str(e)

    return (result, success)

def main():
    audio_path = str(input("Enter the path to the audio file: "))
    model_name = "large-v3"

    result, success = retranscript(audio_path, model_name)

    if success:
        print(f"transcription: {result}\n")
    else:
        print(f"error: {result}\n")

if __name__ == "__main__":
    while True:
        main()