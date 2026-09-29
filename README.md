# Lab of the Future Project with Aria Gen 2 Glasses
This project uses Meta's Aria Gen 2 glasses to create a virtual lab assistant for researchers. 

## Installation
1. Clone the repository
2. Create a Python virtual environment
3. Run `pip install -r requirement.txt`
4. Download the `eat_2` VRS data: `aria_dataset_downloader -c data/eat_2_download_urls.json -o data/ -l all -d 0`
5. [Create a Gemini API key](https://aistudio.google.com/api-keys) and place it in a .env file as `GEMINI_API_KEY=<api_key>`

## Running
Run `python src/main.py` to run the code. For a selected time window, this extracts the relevant VRS data, propagate gazes onto the images, converts audio to text, and generates a VLM description of the scene. The outputs will be stored to `outputs/`.