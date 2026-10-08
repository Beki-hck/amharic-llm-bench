"""Writes the benchmark suites (src/amharic_bench/suites/*.jsonl) from the data below.

Every passage, question and sentence here was written for this benchmark, so it
can't have leaked into a model's training data from an existing dataset. The
passages are fictional. Edit this file, then run `python scripts/build_suites.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "amharic_bench" / "suites"

# ---------------------------------------------------------------------------
# 1. Reading comprehension: passage, then (question, accepted answers, category)
# ---------------------------------------------------------------------------
QA = [
    ("አልማዝ በደሴ ከተማ የምትኖር ነርስ ናት። በየቀኑ ጠዋት ወደ ሆስፒታል በእግሯ ትሄዳለች። ቅዳሜ ቅዳሜ ደግሞ እናቷን በገበያ ትረዳለች። ወንድሟ ዳዊት በአዲስ አበባ መምህር ነው።", [
        ("አልማዝ የምትኖረው በየትኛው ከተማ ነው?", ["ደሴ"], "lookup"),
        ("አልማዝ ሥራዋ ምንድን ነው?", ["ነርስ"], "lookup"),
        ("የአልማዝ ወንድም ሥራው ምንድን ነው?", ["መምህር", "አስተማሪ"], "lookup"),
    ]),
    ("ከበደ ትንሽ የቡና እርሻ አለው። በዚህ ዓመት 300 ኪሎ ቡና ሰብስቧል። ከዚህ ውስጥ ግማሹን ለነጋዴ ሸጦ የቀረውን ለቤተሰቡ አስቀምጧል። ከሽያጩ ባገኘው ገንዘብ ለልጆቹ የትምህርት ቁሳቁስ ገዛ።", [
        ("ከበደ በዚህ ዓመት ስንት ኪሎ ቡና ሰበሰበ?", ["300", "ሦስት መቶ"], "number"),
        ("ከበደ ስንት ኪሎ ቡና ለነጋዴ ሸጠ?", ["150", "መቶ ሃምሳ"], "inference"),
        ("ከበደ በሽያጩ ገንዘብ ምን ገዛ?", ["የትምህርት ቁሳቁስ", "የትምህርት ቁሳቁሶች"], "lookup"),
    ]),
    ("ዓባይ ወንዝ ከጣና ሐይቅ ይነሳል። ወንዙ በሱዳን ውስጥ ከነጭ ዓባይ ጋር ተገናኝቶ ወደ ግብፅ ይፈሳል። ኢትዮጵያ በወንዙ ላይ ታላቁን የኢትዮጵያ ሕዳሴ ግድብ ገንብታለች።", [
        ("ዓባይ ወንዝ የሚነሳው ከየት ነው?", ["ጣና ሐይቅ", "ጣና"], "lookup"),
        ("ዓባይ ከነጭ ዓባይ ጋር የሚገናኘው በየትኛው አገር ነው?", ["ሱዳን"], "lookup"),
        ("ኢትዮጵያ በወንዙ ላይ ምን ገነባች?", ["ሕዳሴ ግድብ", "ግድብ"], "lookup"),
    ]),
    ("ሰላም የሶፍትዌር መሐንዲስ ናት። ባለፈው ወር ለአነስተኛ ሱቆች የሚሆን የሞባይል መተግበሪያ ሠራች። መተግበሪያው ሱቆች ሽያጫቸውን እንዲመዘግቡ ይረዳል። እስካሁን 40 ሱቆች መተግበሪያውን መጠቀም ጀምረዋል።", [
        ("ሰላም የሠራችው መተግበሪያ ለማን የሚሆን ነው?", ["አነስተኛ ሱቆች", "ሱቆች"], "lookup"),
        ("መተግበሪያውን ስንት ሱቆች መጠቀም ጀምረዋል?", ["40", "አርባ"], "number"),
    ]),
    ("ትናንት ከባድ ዝናብ በመዝነቡ ብዙ መንገዶች በውሃ ተጥለቀለቁ። በዚህ የተነሳ ትምህርት ቤቶች ተዘግተው ዋሉ። የከተማው አስተዳደር ዛሬ መንገዶቹን ማጽዳት ጀምሯል።", [
        ("ትምህርት ቤቶች የተዘጉት ለምንድን ነው?", ["ዝናብ", "ጎርፍ", "በውሃ ተጥለቀለቁ", "መንገዶች በውሃ በመጥለቅለቃቸው"], "inference"),
        ("መንገዶቹን ማጽዳት የጀመረው ማን ነው?", ["የከተማው አስተዳደር", "የከተማ አስተዳደር", "አስተዳደሩ"], "lookup"),
    ]),
    ("መሠረት በየቀኑ ስድስት ኪሎ ሜትር ትሮጣለች። በሚቀጥለው ወር በሚካሄደው የሩጫ ውድድር ለመሳተፍ እየተዘጋጀች ነው። አሰልጣኟ ብዙ አትክልት እንድትመገብ እና በቂ እንቅልፍ እንድትተኛ መክሯታል።", [
        ("መሠረት በቀን ስንት ኪሎ ሜትር ትሮጣለች?", ["6", "ስድስት"], "number"),
        ("መሠረት ለምን እየተዘጋጀች ነው?", ["የሩጫ ውድድር", "ውድድር", "ውድድሩ"], "lookup"),
        ("አሰልጣኟ ምን እንድትመገብ መከራት?", ["አትክልት"], "lookup"),
    ]),
    ("የዮሐንስ ቤተሰብ አምስት አባላት አሉት። አባቱ ታክሲ ይነዳል፤ እናቱ ደግሞ ትንሽ ምግብ ቤት አላት። ዮሐንስ እና ታላቅ እህቱ ከትምህርት በኋላ እናታቸውን በምግብ ቤቱ ያግዛሉ። ታናሽ ወንድሙ ገና የአንደኛ ክፍል ተማሪ ነው።", [
        ("የዮሐንስ አባት ሥራው ምንድን ነው?", ["ታክሲ", "ሹፌር", "ታክሲ መንዳት"], "lookup"),
        ("የዮሐንስ ቤተሰብ ስንት አባላት አሉት?", ["5", "አምስት"], "number"),
        ("ዮሐንስ ከትምህርት በኋላ ማንን ያግዛል?", ["እናቱን", "እናቱ", "እናታቸውን"], "lookup"),
    ]),
    ("ላሊበላ በሰሜን ኢትዮጵያ የምትገኝ ታሪካዊ ከተማ ናት። ከተማዋ ከአለት ተፈልፍለው በተሠሩ አብያተ ክርስቲያናት ትታወቃለች። እነዚህ አብያተ ክርስቲያናት በንጉሥ ላሊበላ ዘመን እንደተሠሩ ይነገራል። በየዓመቱ በገና በዓል ብዙ ምዕመናን ወደ ከተማዋ ይጓዛሉ።", [
        ("ላሊበላ በምን ትታወቃለች?", ["አብያተ ክርስቲያናት", "ቤተ ክርስቲያናት", "ቤተክርስቲያናት"], "lookup"),
        ("ብዙ ምዕመናን ወደ ላሊበላ የሚጓዙት በየትኛው በዓል ነው?", ["ገና"], "lookup"),
    ]),
    ("የቢሮው ኮምፒውተር ትናንት መሥራት አቆመ። ቴክኒሻኑ ችግሩ የተበላሸ የኃይል አቅርቦት መሆኑን አገኘ። አዲስ የኃይል አቅርቦት እስኪመጣ ድረስ ሠራተኞቹ ላፕቶፕ ይጠቀማሉ።", [
        ("የኮምፒውተሩ ችግር ምን ነበር?", ["የኃይል አቅርቦት"], "lookup"),
        ("ሠራተኞቹ አዲሱ ዕቃ እስኪመጣ ምን ይጠቀማሉ?", ["ላፕቶፕ"], "lookup"),
    ]),
    ("ሔኖክ በሐዋሳ ሐይቅ አጠገብ ትንሽ ሆቴል ከፍቷል። ሆቴሉ አሥር ክፍሎች አሉት። ብዙ እንግዶች የሚመጡት የአሳ ምግብ ለመብላት እና ሐይቁን ለማየት ነው። በክረምት ወቅት ግን የእንግዶች ቁጥር ይቀንሳል።", [
        ("ሔኖክ ሆቴሉን የከፈተው የት ነው?", ["ሐዋሳ"], "lookup"),
        ("ሆቴሉ ስንት ክፍሎች አሉት?", ["10", "አሥር"], "number"),
        ("የእንግዶች ቁጥር የሚቀንሰው መቼ ነው?", ["ክረምት"], "lookup"),
    ]),
]

# ---------------------------------------------------------------------------
# 2. Multiple choice: (question, correct option, [3 distractors], category).
#    Options are shuffled with a fixed seed so the right letter is spread over A-D.
# ---------------------------------------------------------------------------
MCQ = [
    ("የኢትዮጵያ ዋና ከተማ የትኛዋ ናት?", "አዲስ አበባ", ["ባሕር ዳር", "ጎንደር", "ሐረር"], "culture"),
    ("የኢትዮጵያ የዘመን አቆጣጠር በዓመት ስንት ወራት አሉት?", "13", ["12", "11", "14"], "culture"),
    ("ኢትዮጵያውያን አዲስ ዓመትን (እንቁጣጣሽን) የሚያከብሩት በየትኛው ወር ነው?", "መስከረም", ["ጥር", "ሚያዝያ", "ሐምሌ"], "culture"),
    ("ጤፍ በዋነኝነት ለየትኛው ምግብ ይውላል?", "እንጀራ", ["ቡና", "ጠጅ", "ቅቤ"], "culture"),
    ("አማርኛ የሚጻፍበት ፊደል ምን ተብሎ ይጠራል?", "ግዕዝ (ፊደል)", ["ላቲን", "ዓረብኛ", "ሲሪሊክ"], "culture"),
    ("የዓድዋ ጦርነት የተካሄደው በየትኛው ዓመት (በአውሮፓውያን አቆጣጠር) ነው?", "1896", ["1935", "1941", "1974"], "history"),
    ("በዓድዋ ጦርነት ኢትዮጵያ ያሸነፈችው የትኛውን አገር ጦር ነው?", "ጣሊያን", ["እንግሊዝ", "ፈረንሳይ", "ጀርመን"], "history"),
    ("ከሚከተሉት ውስጥ የኢትዮጵያ ከፍተኛው ተራራ የትኛው ነው?", "ራስ ዳሽን", ["ቱሉ ዲምቱ", "እንጦጦ", "ዝቋላ"], "geography"),
    ("ከሚከተሉት ሐይቆች ውስጥ በኢትዮጵያ የሚገኘው የትኛው ነው?", "ጣና", ["ቪክቶሪያ", "ታንጋኒካ", "ቻድ"], "geography"),
    ("አዲስ አበባ የምትገኘው በየትኛው አህጉር ነው?", "አፍሪካ", ["እስያ", "አውሮፓ", "ደቡብ አሜሪካ"], "geography"),
    ("የኢትዮጵያ ገንዘብ ምን ተብሎ ይጠራል?", "ብር", ["ሺሊንግ", "ናይራ", "ዲናር"], "culture"),
    ("የኢትዮጵያ ሰንደቅ ዓላማ ሦስቱ ዋና ቀለማት የትኞቹ ናቸው?", "አረንጓዴ፣ ቢጫ እና ቀይ", ["ጥቁር፣ ነጭ እና ቀይ", "ሰማያዊ፣ ነጭ እና ጥቁር", "ቀይ፣ ነጭ እና ሰማያዊ"], "culture"),
    ("የጥምቀት በዓል የሚከበረው በየትኛው ወር ነው?", "ጥር", ["መስከረም", "ግንቦት", "ነሐሴ"], "culture"),
    ("የመስቀል በዓል የሚከበረው በየትኛው ወር ነው?", "መስከረም", ["ታኅሣሥ", "የካቲት", "ሰኔ"], "culture"),
    ("ክራር ምንድን ነው?", "የሙዚቃ መሣሪያ", ["የምግብ ዓይነት", "ወንዝ", "ከተማ"], "culture"),
    ("ዶሮ ወጥ ምንድን ነው?", "የባህል ምግብ", ["የመጠጥ ዓይነት", "የሙዚቃ መሣሪያ", "የልብስ ዓይነት"], "culture"),
    ("ከሰኞ ቀጥሎ የሚመጣው ቀን የትኛው ነው?", "ማክሰኞ", ["እሑድ", "ረቡዕ", "ዓርብ"], "vocabulary"),
    ("\"ውሃ\" የሚለው ቃል በእንግሊዝኛ ምን ማለት ነው?", "water", ["fire", "bread", "house"], "vocabulary"),
    ("\"ሰላም\" የሚለው ቃል በእንግሊዝኛ ምን ማለት ነው?", "peace", ["war", "rain", "road"], "vocabulary"),
    ("የ\"ትልቅ\" ተቃራኒ ቃል የትኛው ነው?", "ትንሽ", ["ረጅም", "ሰፊ", "ከባድ"], "vocabulary"),
    ("የ\"ፈጣን\" ተቃራኒ ቃል የትኛው ነው?", "ቀርፋፋ", ["ቀላል", "ረጅም", "አዲስ"], "vocabulary"),
    ("የ\"ልጅ\" ብዙ ቁጥር የትኛው ነው?", "ልጆች", ["ልጅነት", "ልጅቱ", "ልጃገረድ"], "grammar"),
    ("\"ልጆቹ ኳስ ___\" የሚለውን ዓረፍተ ነገር በትክክል የሚያሟላው የትኛው ነው?", "ይጫወታሉ", ["ይጫወታል", "ትጫወታለች", "እጫወታለሁ"], "grammar"),
    ("\"እኔ ትናንት ወደ ገበያ ___\" የሚለውን ዓረፍተ ነገር በትክክል የሚያሟላው የትኛው ነው?", "ሄድኩ", ["ሄደች", "ሄዱ", "ሄድን"], "grammar"),
    ("\"እሷ በየቀኑ ቡና ___\" የሚለውን ዓረፍተ ነገር በትክክል የሚያሟላው የትኛው ነው?", "ትጠጣለች", ["ይጠጣል", "እጠጣለሁ", "ይጠጣሉ"], "grammar"),
    ("አሥራ አምስት ሲደመር ሃያ ሰባት ስንት ነው?", "42", ["32", "41", "52"], "math"),
    ("አንድ ሰው 100 ብር ይዞ 35 ብር ቢያወጣ ስንት ብር ይቀረዋል?", "65", ["75", "55", "135"], "math"),
]

# ---------------------------------------------------------------------------
# 3. Translation pairs (English, [Amharic references], domain). Used in both
#    directions so the en->am vs am->en gap reflects direction, not content.
#    The first Amharic reference is the source text for am->en.
# ---------------------------------------------------------------------------
PAIRS = [
    ("The market opens early on Saturday morning.", ["ገበያው ቅዳሜ ጠዋት በማለዳ ይከፈታል።"], "daily"),
    ("My grandmother makes coffee for the whole family every afternoon.", ["አያቴ በየቀኑ ከሰዓት በኋላ ለመላው ቤተሰብ ቡና ታፈላለች።"], "daily"),
    ("Please drink plenty of water when you have a fever.", ["ትኩሳት ሲኖርዎት እባክዎ ብዙ ውሃ ይጠጡ።", "ትኩሳት ሲኖርህ እባክህ ብዙ ውሃ ጠጣ።"], "health"),
    ("The bus to Bahir Dar is full today.", ["ወደ ባሕር ዳር የሚሄደው አውቶቡስ ዛሬ ሞልቷል።"], "travel"),
    ("Farmers in the region are waiting for the rainy season.", ["በአካባቢው ያሉ ገበሬዎች የክረምቱን ወቅት እየጠበቁ ነው።"], "agriculture"),
    ("She studies computer science at Addis Ababa University.", ["እሷ በአዲስ አበባ ዩኒቨርሲቲ የኮምፒውተር ሳይንስ ትማራለች።"], "education"),
    ("How much does one kilogram of teff cost?", ["አንድ ኪሎ ጤፍ ስንት ነው?", "የአንድ ኪሎ ግራም ጤፍ ዋጋ ስንት ነው?"], "daily"),
    ("The doctor said the child needs to rest for three days.", ["ሐኪሙ ልጁ ለሦስት ቀናት ማረፍ እንዳለበት ተናገረ።"], "health"),
    ("We forgot to close the window before the storm.", ["ከአውሎ ነፋሱ በፊት መስኮቱን መዝጋት ረሳን።"], "daily"),
    ("Our team won the football match yesterday.", ["ቡድናችን ትናንት የእግር ኳስ ጨዋታውን አሸነፈ።"], "sports"),
    ("Can you help me carry these bags?", ["እነዚህን ቦርሳዎች እንድሸከም ልትረዳኝ ትችላለህ?", "እነዚህን ቦርሳዎች ለመሸከም ሊረዱኝ ይችላሉ?"], "daily"),
    ("The new road has reduced the travel time between the two towns.", ["አዲሱ መንገድ በሁለቱ ከተሞች መካከል ያለውን የጉዞ ጊዜ ቀንሷል።"], "travel"),
    ("Wash your hands with soap before eating.", ["ከመብላትዎ በፊት እጅዎን በሳሙና ይታጠቡ።", "ከመብላትህ በፊት እጅህን በሳሙና ታጠብ።"], "health"),
    ("The library is closed on public holidays.", ["ቤተ መጻሕፍቱ በሕዝባዊ በዓላት ቀን ዝግ ነው።"], "education"),
    ("He sent the money to his brother using his phone.", ["ገንዘቡን በስልኩ ተጠቅሞ ለወንድሙ ላከ።"], "business"),
    ("Lake Tana is the largest lake in Ethiopia.", ["ጣና ሐይቅ በኢትዮጵያ ውስጥ ትልቁ ሐይቅ ነው።"], "geography"),
    ("I have been learning to drive for two months.", ["መኪና መንዳት መማር ከጀመርኩ ሁለት ወር ሆኖኛል።"], "daily"),
    ("The price of fuel increased again this week.", ["የነዳጅ ዋጋ በዚህ ሳምንት እንደገና ጨመረ።"], "business"),
    ("Children should get enough sleep every night.", ["ልጆች በየምሽቱ በቂ እንቅልፍ ማግኘት አለባቸው።"], "health"),
    ("The meeting was postponed until next Monday.", ["ስብሰባው እስከሚቀጥለው ሰኞ ድረስ ተራዘመ።", "ስብሰባው ወደሚቀጥለው ሰኞ ተላልፏል።"], "business"),
    ("This medicine should be taken twice a day after meals.", ["ይህ መድኃኒት በቀን ሁለት ጊዜ ከምግብ በኋላ መወሰድ አለበት።"], "health"),
    ("The students planted trees around the school.", ["ተማሪዎቹ በትምህርት ቤቱ ዙሪያ ዛፎችን ተከሉ።"], "education"),
    ("My phone battery is almost dead.", ["የስልኬ ባትሪ ሊያልቅ ነው።"], "daily"),
    ("According to tradition, coffee was first discovered in Kaffa.", ["በአፈ ታሪክ መሠረት ቡና መጀመሪያ የተገኘው በከፋ ነው።"], "culture"),
    ("Thank you for inviting us to the wedding.", ["ወደ ሰርጉ ስለጋበዛችሁን እናመሰግናለን።", "ወደ ሰርጉ ስለጋበዙን እናመሰግናለን።"], "daily"),
]

# ---------------------------------------------------------------------------
# 4. Summarization: (passage, one-sentence reference summary, topic)
# ---------------------------------------------------------------------------
SUMMARIES = [
    ("የአዲስ አበባ ከተማ አስተዳደር በከተማዋ ውስጥ አዳዲስ የሕዝብ አውቶቡሶችን ሥራ ላይ አዋለ። አውቶቡሶቹ በዋና ዋና መንገዶች ላይ በየአሥር ደቂቃው ይመላለሳሉ። ባለሥልጣናት እንዳሉት ይህ እርምጃ የትራንስፖርት ችግርን ለመቀነስ ያለመ ነው። ተሳፋሪዎች በአዲሱ አገልግሎት መደሰታቸውን ገልጸዋል።",
     "የአዲስ አበባ አስተዳደር የትራንስፖርት ችግርን ለመቀነስ አዳዲስ የሕዝብ አውቶቡሶችን ሥራ ላይ አዋለ።", "transport"),
    ("በዚህ ዓመት በአማራ ክልል የዘነበው ዝናብ ከወትሮው የበለጠ ነበር። በዚህም የተነሳ የጤፍ እና የስንዴ ምርት ጨምሯል። ገበሬዎቹ ምርታቸውን በጥሩ ዋጋ መሸጣቸውን ተናግረዋል። ሆኖም አንዳንድ አካባቢዎች በጎርፍ ተጎድተዋል።",
     "በአማራ ክልል ከወትሮው የበለጠ ዝናብ በመዝነቡ የጤፍና የስንዴ ምርት ጨመረ፤ አንዳንድ አካባቢዎች ግን በጎርፍ ተጎድተዋል።", "agriculture"),
    ("የጤና ባለሙያዎች ሕፃናት በወቅቱ ክትባት እንዲወስዱ ወላጆችን አሳሰቡ። ክትባት እንደ ኩፍኝ ያሉ በሽታዎችን ለመከላከል ይረዳል። በዚህ ወር በመላ አገሪቱ ነፃ የክትባት ዘመቻ ይካሄዳል። ወላጆች ልጆቻቸውን በአቅራቢያቸው ወዳለ ጤና ጣቢያ እንዲወስዱ ተጠይቀዋል።",
     "የጤና ባለሙያዎች ወላጆች በዚህ ወር በሚካሄደው ነፃ የክትባት ዘመቻ ልጆቻቸውን እንዲያስከትቡ አሳሰቡ።", "health"),
    ("አንድ የኢትዮጵያ ጀማሪ ኩባንያ ገበሬዎች በስልካቸው የአየር ሁኔታ መረጃ የሚያገኙበትን አገልግሎት ጀመረ። አገልግሎቱ መረጃውን በአማርኛ እና በኦሮምኛ በአጭር የጽሑፍ መልእክት ይልካል። ኩባንያው እስካሁን ከአሥር ሺህ በላይ ገበሬዎች መመዝገባቸውን ገልጿል።",
     "አንድ የኢትዮጵያ ጀማሪ ኩባንያ ለገበሬዎች የአየር ሁኔታ መረጃን በአማርኛና በኦሮምኛ በስልክ መልእክት የሚልክ አገልግሎት ጀመረ።", "technology"),
    ("አንዲት ኢትዮጵያዊት አትሌት በበርሊን ማራቶን አንደኛ ወጣች። ውድድሩን ለማጠናቀቅ ሁለት ሰዓት ከአሥራ ዘጠኝ ደቂቃ ፈጅቶባታል። ይህ በሙያ ዘመኗ ያስመዘገበችው ምርጥ ሰዓት ነው። አትሌቷ ድሉን ለቤተሰቧና ለአሰልጣኟ መስጠቷን ተናግራለች።",
     "አንዲት ኢትዮጵያዊት አትሌት በሙያ ዘመኗ ምርጥ በሆነ ሰዓት የበርሊን ማራቶንን በአንደኝነት አጠናቀቀች።", "sports"),
    ("የትምህርት ሚኒስቴር የሁለተኛ ደረጃ ትምህርት ቤቶች የኮምፒውተር ትምህርት እንዲሰጡ ወሰነ። ለዚህም ለትምህርት ቤቶች ኮምፒውተሮች ይከፋፈላሉ። መምህራንም ልዩ ሥልጠና ይሰጣቸዋል። ውሳኔው ከሚቀጥለው የትምህርት ዘመን ጀምሮ ተግባራዊ ይሆናል።",
     "የትምህርት ሚኒስቴር ከሚቀጥለው የትምህርት ዘመን ጀምሮ የሁለተኛ ደረጃ ትምህርት ቤቶች የኮምፒውተር ትምህርት እንዲሰጡ ወሰነ።", "education"),
    ("በደቡብ ኢትዮጵያ የሚኖሩ ወጣቶች በአካባቢያቸው አንድ ሚሊዮን ችግኞችን ለመትከል ዘመቻ ጀመሩ። ዓላማቸው የደን መመናመንን መከላከል እና የአፈር መሸርሸርን መቀነስ ነው። የአካባቢው ነዋሪዎችም በዘመቻው በንቃት እየተሳተፉ ነው።",
     "በደቡብ ኢትዮጵያ ያሉ ወጣቶች የደን መመናመንን ለመከላከል አንድ ሚሊዮን ችግኞችን የመትከል ዘመቻ ጀመሩ።", "environment"),
    ("የቡና ዋጋ በዓለም ገበያ በመጨመሩ የኢትዮጵያ የቡና ወጪ ንግድ ገቢ አደገ። ኢትዮጵያ ቡናዋን በዋናነት ወደ ጀርመን፣ ሳዑዲ ዓረቢያ እና አሜሪካ ትልካለች። ላኪዎች ግን የትራንስፖርት ወጪ መጨመሩ ትርፋቸውን እንደቀነሰው ገልጸዋል።",
     "የቡና ዋጋ በዓለም ገበያ በመጨመሩ የኢትዮጵያ የቡና ወጪ ንግድ ገቢ ቢያድግም የትራንስፖርት ወጪ የላኪዎችን ትርፍ ቀንሷል።", "business"),
    ("በሰፈራችን የነበረው የውሃ ችግር ከሁለት ሳምንት በኋላ ተፈታ። ነዋሪዎቹ ውሃ ከሩቅ ቦታ ለማምጣት ተገደው ነበር። የውሃ ባለሥልጣኑ የተበላሸውን ቧንቧ ጠግኖ አገልግሎቱን መልሷል።",
     "የውሃ ባለሥልጣኑ የተበላሸውን ቧንቧ በመጠገኑ በሰፈሩ ለሁለት ሳምንት የቆየው የውሃ ችግር ተፈታ።", "community"),
    ("የመስቀል በዓል በመላው ኢትዮጵያ በድምቀት ተከበረ። በአዲስ አበባ መስቀል አደባባይ በሺዎች የሚቆጠሩ ሰዎች ተገኝተዋል። በዓሉ የሚከበረው ደመራ በማብራት ነው። በዓሉ በዩኔስኮ የማይዳሰስ ቅርስ ሆኖ ተመዝግቧል።",
     "በዩኔስኮ የተመዘገበው የመስቀል በዓል ደመራ በማብራት በመላው ኢትዮጵያ በድምቀት ተከበረ።", "culture"),
]

SYSTEMS = {
    "am_qa": "ከታች ያለውን ጽሑፍ ያንብቡና ጥያቄውን ይመልሱ። መልሱን በጥቂት ቃላት፣ በአማርኛ ብቻ ይጻፉ።",
    "am_mcq": "ለጥያቄው ትክክለኛውን አማራጭ ይምረጡ። የመጨረሻውን መስመር 'መልስ: <ፊደል>' በሚል ቅርጽ ይጻፉ (ለምሳሌ፦ መልስ: B)።",
    "en_am": "You are a professional translator. Translate the user's English text into Amharic. Reply with the Amharic translation only.",
    "am_en": "You are a professional translator. Translate the user's Amharic text into English. Reply with the English translation only.",
    "am_sum": "የተሰጠውን ጽሑፍ በአንድ ዓረፍተ ነገር በአማርኛ ያጠቃልሉ። ማጠቃለያውን ብቻ ይጻፉ።",
}


def build() -> dict[str, list[dict]]:
    suites: dict[str, list[dict]] = {k: [] for k in SYSTEMS}

    n = 0
    for p, (passage, questions) in enumerate(QA, 1):
        for question, answers, cat in questions:
            n += 1
            suites["am_qa"].append({
                "id": f"qa-{n:02d}", "category": cat, "passage": p, "scorer": "am_qa", "expected": answers,
                "prompt": f"ጽሑፍ፦ {passage}\n\nጥያቄ፦ {question}",
            })

    rng = random.Random(1896)  # fixed seed: the same letters every build
    for i, (question, right, wrong, cat) in enumerate(MCQ, 1):
        options = [right, *wrong]
        rng.shuffle(options)
        letters = "ABCD"
        body = "\n".join(f"{letters[j]}) {o}" for j, o in enumerate(options))
        suites["am_mcq"].append({
            "id": f"mcq-{i:02d}", "category": cat, "scorer": "am_choice",
            "expected": letters[options.index(right)], "prompt": f"{question}\n\n{body}",
        })

    for i, (en, ams, domain) in enumerate(PAIRS, 1):
        suites["en_am"].append({
            "id": f"en-am-{i:02d}", "category": domain, "scorer": "chrf", "target": "am",
            "expected": ams, "prompt": f"Translate into Amharic:\n\n{en}",
        })
        suites["am_en"].append({
            "id": f"am-en-{i:02d}", "category": domain, "scorer": "chrf", "target": "en",
            "expected": [en], "prompt": f"Translate into English:\n\n{ams[0]}",
        })

    for i, (passage, ref, topic) in enumerate(SUMMARIES, 1):
        suites["am_sum"].append({
            "id": f"sum-{i:02d}", "category": topic, "scorer": "chrf", "target": "am", "pass_chrf": 35,
            "expected": [ref], "prompt": f"ጽሑፍ፦ {passage}",
        })
    return suites


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in build().items():
        lines = [json.dumps({"system": SYSTEMS[name]}, ensure_ascii=False)]
        lines += [json.dumps(it, ensure_ascii=False) for it in items]
        (OUT / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(items)} items")


if __name__ == "__main__":
    main()
