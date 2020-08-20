#parser for subtitles lol
#https://subscene.com
#DO NOT REDISTRIBUTE compiled files or subtitles - I DO NOT OWN THEM.
import os
import re
import random
from tqdm import trange
endfile  = open("input1.txt", "w")
lsdir = os.listdir("Extracted")
random.shuffle(lsdir)
t = trange(len(lsdir), desc='Scanning files', leave=True)

for filename in lsdir:
    if filename.endswith(".ass"): 
        t.update(1)
        qbfile = open("Extracted/"+filename, "r", encoding="utf-8", errors='ignore')
        for readline in qbfile:
            readline= re.sub('{[^>]+}', '', readline)
            readline=readline.replace(r"\N", " ")
            readline=readline.replace("–", "-")
            readline=readline.replace("…", "")
            readline=readline.replace("—", "-")
            readline=readline.replace("é", "e")
            readline=readline.replace("è", "e")
            readline=readline.replace("‘", "'")
            readline=readline.replace("’", "'")
            readline=readline.replace("“", '"')
            readline=readline.replace("”", '"')
            readline=readline.replace("¥", 'Y')
            readline=readline.replace("°", '')
            if str(readline).startswith("Dialogue:"):  
                #print(readline)
                splitline=readline.split(",")
                if len(splitline) == 10:
                    try:
                        endfile.write(">"+splitline[9].replace("\n", " ")[:-1] + "\n")
                    except:
                        pass
                elif len(splitline) > 10:
                    comb=""
                    for i in range(9, len(splitline)):
                        if i < len(splitline)-1:
                            comb=comb+splitline[i]+","
                        else:
                            comb=comb+splitline[i]
                    try:
                        endfile.write(">"+comb.replace("\n", " ")[:-1] + "\n")
                    except:
        
                        pass
        qbfile.close()
    elif filename.endswith(".srt"):
        t.update(1)
        qbfile = open("Extracted/"+filename, "r", encoding="ISO-8859-1")
        combline=""
        for readline in qbfile:
            readline= re.sub('<[^>]+>', '', readline)
            readline= re.sub('[[^>]+]', '', readline)
            readline=readline.replace("–", "-")
            readline=readline.replace("…", "")
            readline=readline.replace("—", "-")
            readline=readline.replace("é", "e")
            readline=readline.replace("è", "e")
            readline=readline.replace("‘", "'")
            readline=readline.replace("’", "'")
            readline=readline.replace("“", '"')
            readline=readline.replace("”", '"')
            readline=readline.replace("¥", 'Y')
            readline=readline.replace("°", '')
            if readline=="\n":
                combline.replace("\n", " ")
                splitline=combline.split(" ")
                if len(splitline) == 5:
                    try:
                        endfile.write(">"+splitline[4].replace("\n", " ")[:-1] + "\n")
                    except:
                        pass
                elif len(splitline) > 5:
                    comb=""
                    for i in range(4, len(splitline)):
                        if i < len(splitline)-1:
                            comb=comb+splitline[i]+" "
                        else:
                            comb=comb+splitline[i]
                    try:
                        endfile.write(r">"+comb.replace("\n", " ")[:-1] + "\n")
                    except:
                        pass
                combline=""
            combline=combline+readline

        qbfile.close()

endfile.close()

listd=[">yu",">me",">ka",">na",">ru",">ta",">bi",">ko",">wa",">re",">te",">ku",">ze",">so",
        ">de",">mo",">ma",">bo",">sa",">su",">ha",">ri",">tsu",">zu",">ke",">bu",">atta",
        ">fu",">ki",">watta?",">da",">rou",">ho",">ro",">mou",">to",">ga",">ni",">kyo",
        ">ra",">tto...",">do",">chi",">mi",">shi",">ya",">ba",">sho",">he",">ai",">ji",
        ">mu",">ne",">pe",">yo",">he",">se"]

newfile=open("input.txt", "w")
oldfile=open("input1.txt", "r", encoding="utf-8", errors='ignore')
print("\nStart cleaning")
lastline0=""
lastline1=""
lastline2=""
lastline3=""
lastline4=""
lastline5=""
lastline6=""
lastline7=""
fullread=[]
for line in oldfile:
    fullread.append(line)
for index,readline in enumerate(fullread):
    if readline==">\n":
        pass
    elif readline==lastline0 or readline==lastline1 or readline==lastline2 or readline==lastline3:
        lastline7=lastline6
        lastline6=lastline5
        lastline5=lastline4
        lastline4=lastline3
        lastline3=lastline2
        lastline2=lastline1
        lastline1=lastline0
        lastline0=readline
        pass
    elif readline==lastline4 or readline==lastline5 or readline==lastline6 or readline==lastline7:
        lastline7=lastline6
        lastline6=lastline5
        lastline5=lastline4
        lastline4=lastline3
        lastline3=lastline2
        lastline2=lastline1
        lastline1=lastline0
        lastline0=readline
        pass
    elif len(readline.replace("\n",""))<=2:
        lastline7=lastline6
        lastline6=lastline5
        lastline5=lastline4
        lastline4=lastline3
        lastline3=lastline2
        lastline2=lastline1
        lastline1=lastline0
        lastline0=readline
        pass
    elif len(readline.replace("\n",""))>=370:
        lastline7=lastline6
        lastline6=lastline5
        lastline5=lastline4
        lastline4=lastline3
        lastline3=lastline2
        lastline2=lastline1
        lastline1=lastline0
        lastline0=readline
        pass
    elif readline.replace("\n","").startswith(">m "):
        lastline7=lastline6
        lastline6=lastline5
        lastline5=lastline4
        lastline4=lastline3
        lastline3=lastline2
        lastline2=lastline1
        lastline1=lastline0
        lastline0=readline
        pass
    else:
        lastline7=lastline6
        lastline6=lastline5
        lastline5=lastline4
        lastline4=lastline3
        lastline3=lastline2
        lastline2=lastline1
        lastline1=lastline0
        lastline0=readline
        rmn = readline.replace("\n", "")
        
        try: 
            if fullread[index+1].startswith(">...") and readline.endswith("...\n"):
                #NOT DONE, FIX!!!!
                newfile.write((str(readline.replace("\n", " "))+fullread[index+1]).replace("> ", ">").replace("  ", " ").replace("\n", " ").replace("... >...", "")[:-1] + "\n")
                lastline0=fullread[index+1]
            else:
                found_err=False
                for errorword in listd:
                    if errorword in readline.replace("\n", ""):
                        found_err=True
                if found_err==False:
                    newfile.write(str(readline).replace("\n", " ")[:-1] + "\n")
        except:
            pass
newfile.close()
oldfile.close()