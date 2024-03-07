import gc
import whisper

def retranscript(audio_path, model_name="base"):
    global success
    success = False

    try:
        modelLoaded = whisper.load_model(name=model_name)
        result = modelLoaded.transcribe(audio_path)

        del modelLoaded
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