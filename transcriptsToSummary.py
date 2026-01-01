from openai import OpenAI
import os
import glob
from dotenv import load_dotenv

load_dotenv()
client = OpenAI() #No need for OpenAI(api_key=os.getenv("OPENAI_API_KEY")) because already loaded with dotenv

def generate_chatgpt_response(prompt):
    response = client.chat.completions.create(
    model=os.getenv("SUMMARY_MODEL"),
    messages=[
        {"role": "system", "content": "You are a summarizer of Role Play Game, that take the audio noisy transcript and turn it into stories. The langauge to write is "+os.getenv("TARGET_LANG")+"."},
        {"role": "user", "content": prompt}
    ],
    )
    return response.choices[0].message.content

# Usage example
def transcriptsToSummary():
    print("Starting transcriptsToSummary...")
    with open(os.getenv("CONTEXT"), 'r', encoding='utf-8') as file:
        context = file.read()
    with open(os.getenv("SUMMARY_IN"), 'r', encoding='utf-8') as file:
        summary = file.read()

    #var: previously
    files = glob.glob(os.path.join(os.getenv("SUMMARIES_FOLDER"), "*"))
    if not files:
        print ("WARNING: no previously in " + os.getenv("SUMMARIES_FOLDER"))
        previously = "No previously yet. Just use the transcript, considering this is the start of adventure."
    else: 
        with open(max(files, key=os.path.getmtime), 'r', encoding='utf-8') as file:
            previously = file.read()
            #print(previously)

    #var: lasttranscripts
    files = glob.glob(os.path.join(os.getenv("TRANSCRIPTS_DIR"), "*"))
    if not files:
        print ("ERROR: no last transcripts in " + os.getenv("TRANSCRIPTS_DIR"))
        return
    else:
        with open(max(files, key=os.path.getmtime), 'r', encoding='utf-8') as file:
            #var: ouput_name
            base_filename = os.path.basename(file.name)
            short_name = base_filename[:15]
            output_name = os.path.join(os.getenv("SUMMARIES_FOLDER"), f"{short_name}_summary.txt")
            #output_name = os.getenv("SUMMARIES_FOLDER") + file.name.split("\\")[-1][:15]+"_summary.txt"
            lasttranscripts = file.read()

    with open(os.getenv("PROMPT"), 'r', encoding='utf-8') as file:
        prompt = file.read()

    prompt = prompt + f"""\n
    \n
    **Additional data (context, campaign summary, previous summary, and the transcript to be transformed)**\n
    **Game context:**\n
    {context}\n
    \n
    **Summary of the start of the campaign:**\n
    {summary}\n
    \n
    **Notes taken from the previous session:**\n
    {previously}\n
    \n
    **Raw transcript to be transformed into a fantasy tale:**\n
    {lasttranscripts}
    """

    print("Sending clean transcript to gpt.")
    chatgpt_response = generate_chatgpt_response(prompt)

    print("Writting the summary.")
    with open(output_name, "w+", encoding='utf-8') as f:
        f.write(chatgpt_response)
    print("transcriptsToSummary done!")


if __name__ == "__main__":
    transcriptsToSummary()
