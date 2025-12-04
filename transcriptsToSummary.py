from openai import OpenAI
import os
import glob
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI() #No need for OpenAI(api_key=os.getenv("OPENAI_API_KEY")) because already loaded with dotenv

def generate_chatgpt_response(prompt):
    response = client.chat.completions.create(
    model=os.getenv("SUMMARY_MODEL"),
    messages=[
        {"role": "system", "content": "Vous êtes un résumeurs d'histoires fantastiques médiévales."},
        {"role": "user", "content": prompt}
    ],
    )
    return response.choices[0].message.content

"""def clean_seavoicesfiles(seavoicestxt : str):
    #Remove summary at the beginning
    seavoicestxt = "\n".join(seavoicestxt.splitlines()[10:])

    #Enlever les lignes avec par SousTitreur.com
    seavoicestxt = "\n".join(x for x in seavoicestxt.splitlines() if "par SousTitreur.com" not in x)

    #Remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    seavoicestxt = emoji_pattern.sub(r'', seavoicestxt) # no emoji

    #Remove timestamp
    seavoicestxt = "\n".join([line[26:] for line in seavoicestxt.splitlines()])

    return (seavoicestxt)"""

# Usage example
def transcriptsToSummary():
    print("Starting seavoicesToSummary...")
    with open(os.getenv("CONTEXT"), 'r', encoding='utf-8') as file:
        context = file.read()
    with open(os.getenv("SUMMARY"), 'r', encoding='utf-8') as file:
        summary = file.read()

    #var: previously
    files = glob.glob(os.path.join(os.getenv("SUMMARIES_FOLDER"), "*"))
    if not files:
        print ("ERROR: no previously in " + os.getenv("SUMMARIES_FOLDER"))
        return
    else:
        with open(max(files, key=os.path.getmtime), 'r', encoding='utf-8') as file:
            previously = file.read()

    #var: lasttranscripts
    files = glob.glob(os.path.join(os.getenv("TRANSCRIPTS_FOLDER"), "*"))
    if not files:
        print ("ERROR: no last transcripts in " + os.getenv("TRANSCRIPTS_FOLDER"))
        return
    else:
        with open(max(files, key=os.path.getmtime), 'r', encoding='utf-8') as file:
            #var: ouput_name
            output_name = file[:15]+"_summary.txt"
            lasttranscripts = file.read()


    #Parfois ailouros et Salazar utilise le même micro et sont tous les deux identifiés comme Salazar.
    prompt = f"""
    Tâche :
    Transforme le transcript ci-dessous en un récit de fantasy complet et immersif, comme si tu racontais un roman ou une chronique épique. 
    A la fin, rajoute une partie détaillant les noms de tous les personnages et leurs rôles ainsi que les quêtes qui ont été déclenchées ou finies.

    Consignes suivantes à suivre rigoureusement :

    Utilise le contexte du jeu, les notes de la session précédente et le résumé de début de campagne pour reconstruire fidèlement les événements, les noms des personnages, lieux, factions, objets et quêtes.
    Base toi sur la partie d'avant pour continuer l'histoire sans discontinuité sauf elipse temporelle.
    Radblue est le maître du jeu, considère-le comme une voixoff narrant l'histoire mais il ne doit pas apparaître.
    N'écrit que l'histoire racontée par les joueurs, n'inclus pas les dicussions extérieurs à la partie. 
    Le transcript est en plein milieu de partie, il commence à partir des fins des parties précédentes et n'atteint pas nécéssairement la conclusion.
    Seuls les moments du transcript doivent être décris en respectant la chronologie de l'histoire.

    Le transcript est tiré d'un enregistrement audio de mauvaise qualité. Il contient des erreurs, des mots mal retranscrits, des phrases dans le désordre, et de nombreuses digressions hors-jeu :
    - Ignore tout ce qui ne concerne pas directement la narration de la partie.
    - Reconstitue l'ordre logique des événements même si l'audio est confus.
    - Corrige les incohérences et interprète intelligemment les échanges.

    Ton objectif est de produire un récit complet, détaillé et fidèle, sous la forme d'un conte ou d'un roman fantastique :
    Utilise un style littéraire fluide, immersif et descriptif.
    Décris chaque scène de manière vivante : actions, dialogues, émotions, décors, rebondissements.
    Garde tous les noms et éléments importants du lore.
    N'omets aucun événement significatif, même s'il semble confus dans le transcript.
    Le récit peut faire plusieurs milliers de mots si nécessaire. N'hésite pas à développer chaque étape.
    Structure le récit avec des paragraphes narratifs clairs. Ne rajoute pas de chapitre, juste un recit de plusieurs paragraphes par moment.

    Ne fais pas de résumé. Ne saute rien d'important. Ne donne aucun avis ou explication. Raconte uniquement l'histoire, comme si tu étais le chroniqueur officiel de cette campagne.

    Après le récit que tu as transformé depuis le transcript, rajoute une partie pour détaillers SEULEMENT ET SEULEMENT EUX, les personnages apparu dans ce scénario (joueurs et non-joueurs) en étant exclusif, en incluant un résumé de leur rôle dans le scénario (juste ce qui se passe avec les personnage dans ce scénario, pas les autres actions, juste ceux qui ont participé aux scénario et ce qu'ils ont fait, rien d'autre).
    Ensuite, après le récit et la liste des personnages, rajoute la liste des grandes actions trouvées ou finies SEULEMENT ET SEULEMENT dans ce transcript et seulement dans la partie en cours, pas les autres précédentes et qui n'ont pas été évoqués hors de la partie actuelle. ("Ce [nom de personnage] demande aux héros d'aller chercher...", "Un monstre menace Phandeline...", "La troupe doit aller chercher de l'aide ou un objet...")

    Données supplémentaires (contexte, résumé de la campagne, résumé précédent, et le transcript à transformer)
    Contexte du jeu :
    {context}

    Résumé du début de la campagne :
    {summary}

    Prise de notes de la session précédente :
    {previously}

    Transcript brut à transformer en conte fantastique :
    {lasttranscripts}
    """
    
    #print("Cleaning transcript.")
    #seavoices = clean_seavoicesfiles(seavoices)
    print("Sending clean transcript to gpt.")
    chatgpt_response = generate_chatgpt_response(prompt)

    print("Ecriture de la réponse.")
    with open(output_name, "w+", encoding='utf-8') as f:
        f.write(chatgpt_response)
    print("transcriptsToSummary done!")


if __name__ == "__main__":
    transcriptsToSummary()