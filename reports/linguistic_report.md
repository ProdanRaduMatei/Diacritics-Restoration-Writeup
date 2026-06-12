# Linguistic Support Report

spaCy is used here as an auxiliary analysis layer, not as the core model.
RoWordNet/NLP-Cube hooks are left optional because they are not required for on-prem inference.

## High-Ambiguity Forms From Corpus

| base | total | forms | entropy |
|---|---:|---|---:|
| evidenta | 5 | evidentă:2, evidența:1, evidență:1, evidenta:1 | 1.92 |
| revarsa | 8 | revărsă:3, revarsa:2, revarsă:2, revărsa:1 | 1.91 |
| tari | 3 | tari:1, tări:1, țari:1 | 1.58 |
| scapa | 5 | scapă:2, scăpa:2, scăpă:1 | 1.52 |
| bunavointa | 6 | bunăvoință:3, bunăvoința:2, bunavoința:1 | 1.46 |
| lasa | 25 | lasă:11, lăsă:11, lăsa:3 | 1.41 |
| cauta | 5 | caută:3, cauta:1, căuta:1 | 1.37 |
| apara | 5 | apără:3, apară:1, apăra:1 | 1.37 |
| prezenta | 12 | prezența:6, prezenta:5, prezentă:1 | 1.33 |
| importanta | 13 | importanță:8, importanța:4, importantă:1 | 1.24 |
| fata | 99 | fața:60, față:34, fata:5 | 1.18 |
| cauza | 6 | cauză:3, cauza:3 | 1.00 |
| sl | 6 | sl:3, șl:3 | 1.00 |
| prada | 6 | prada:3, pradă:3 | 1.00 |
| impiedica | 6 | împiedică:3, împiedica:3 | 1.00 |
| mila | 4 | milă:2, mila:2 | 1.00 |
| sora | 4 | soră:2, sora:2 | 1.00 |
| briza | 4 | briza:2, briză:2 | 1.00 |
| proasta | 4 | proastă:2, proasta:2 | 1.00 |
| placa | 4 | placa:2, placă:2 | 1.00 |
| plimba | 4 | plimbă:2, plimba:2 | 1.00 |
| judeca | 4 | judeca:2, judecă:2 | 1.00 |
| camasa | 4 | cămașa:2, cămașă:2 | 1.00 |
| statistica | 4 | statistică:2, statistica:2 | 1.00 |
| privinta | 4 | privință:2, privința:2 | 1.00 |
| freca | 4 | freca:2, frecă:2 | 1.00 |
| socoteala | 4 | socoteală:2, socoteala:2 | 1.00 |
| dreapta | 4 | dreapta:2, dreaptă:2 | 1.00 |
| crete | 4 | crete:2, crețe:2 | 1.00 |
| presa | 4 | presa:2, presă:2 | 1.00 |

## spaCy Analysis For Hard Cases

### `Am pus peste peste legume.`

Expected: `Am pus pește peste legume.`

The first `peste` is a noun and the second is a preposition; corpus frequency alone prefers `peste`.

| token | lemma | pos | morph | dep |
|---|---|---|---|---|
| Am | avea | AUX | Person=1 | aux |
| pus | pune | VERB | Gender=Masc\|Number=Sing\|VerbForm=Part | ROOT |
| pește | pește | NOUN | Definite=Ind\|Gender=Masc\|Number=Sing | obj |
| peste | peste | ADP | AdpType=Prep\|Case=Acc | case |
| legume | legumă | NOUN | Definite=Ind\|Gender=Fem\|Number=Plur | nmod |
| . | . | PUNCT |  | punct |

### `Fata sta in fata casei.`

Expected: `Fata stă în fața casei.`

`fata` can be `fata`, `fată`, `fața`; the second position is forced by the prepositional phrase.

| token | lemma | pos | morph | dep |
|---|---|---|---|---|
| Fata | fată | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | nsubj |
| stă | sta | AUX | Mood=Ind\|Number=Sing\|Person=3\|Tense=Pres\|VerbForm=Fin | ROOT |
| în | în | ADP | AdpType=Prep\|Case=Acc | case |
| fața | față | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | fixed |
| casei | casă | NOUN | Case=Dat,Gen\|Definite=Def\|Gender=Fem\|Number=Sing | obl |
| . | . | PUNCT |  | punct |

### `Mana dreapta era in mana medicului.`

Expected: `Mâna dreaptă era în mâna medicului.`

`mana` differs between indefinite `mână` and definite `mâna`; local syntax matters.

| token | lemma | pos | morph | dep |
|---|---|---|---|---|
| Mâna | mână | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | nsubj |
| dreaptă | drept | ADJ | Case=Acc,Nom\|Definite=Ind\|Degree=Pos\|Gender=Fem\|Number=Sing | amod |
| era | fi | AUX | Mood=Ind\|Number=Sing\|Person=3\|Tense=Imp\|VerbForm=Fin | cop |
| în | în | ADP | AdpType=Prep\|Case=Acc | case |
| mâna | mână | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | ROOT |
| medicului | medic | NOUN | Case=Dat,Gen\|Definite=Def\|Gender=Masc\|Number=Sing | nmod |
| . | . | PUNCT |  | punct |

### `Para dulce nu e acelasi lucru cu o para.`

Expected: `Para dulce nu e același lucru cu o pară.`

`para` may stay plain in named/foreign contexts, but fruit needs `pară`.

| token | lemma | pos | morph | dep |
|---|---|---|---|---|
| Para | pară | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | nsubj |
| dulce | dulce | ADJ | Case=Acc,Nom\|Definite=Ind\|Degree=Pos\|Gender=Fem\|Number=Sing | amod |
| nu | nu | PART | Polarity=Neg | advmod |
| e | fi | AUX | Mood=Ind\|Number=Sing\|Person=3\|Tense=Pres\|VerbForm=Fin | cop |
| același | același | DET | Case=Acc,Nom\|Gender=Masc\|Number=Sing\|Person=3\|Position=Prenom\|PronType=Dem | det |
| lucru | lucru | NOUN | Definite=Ind\|Gender=Masc\|Number=Sing | ROOT |
| cu | cu | ADP | AdpType=Prep\|Case=Acc | case |
| o | un | DET | Case=Acc,Nom\|Gender=Fem\|Number=Sing\|PronType=Ind | det |
| pară | pară | NOUN | Case=Acc,Nom\|Definite=Ind\|Gender=Fem\|Number=Sing | nmod |
| . | . | PUNCT |  | punct |

### `Casa veche nu era acasa.`

Expected: `Casa veche nu era acasă.`

`casa/casă` and `acasă` show that token frequency is not enough; phrase role is important.

| token | lemma | pos | morph | dep |
|---|---|---|---|---|
| Casa | casă | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | nsubj |
| veche | vechi | ADJ | Case=Acc,Nom\|Definite=Ind\|Degree=Pos\|Gender=Fem\|Number=Sing | amod |
| nu | nu | PART | Polarity=Neg | advmod |
| era | fi | AUX | Mood=Ind\|Number=Sing\|Person=3\|Tense=Imp\|VerbForm=Fin | ROOT |
| acasă | acasă | ADV | Degree=Pos | advmod |
| . | . | PUNCT |  | punct |

### `Sa-si ia cartea sau sa o lase?`

Expected: `Să-și ia cartea sau să o lase?`

Hyphenated clitics test whether the system handles punctuation-adjacent words.

| token | lemma | pos | morph | dep |
|---|---|---|---|---|
| Să | să | PART | Mood=Sub | mark |
| -și | sine | PRON | Case=Dat\|Person=3\|PronType=Prs\|Reflex=Yes\|Strength=Weak\|Variant=Short | dep |
| ia | lua | AUX | Mood=Sub\|Person=3\|Tense=Pres\|VerbForm=Fin | ROOT |
| cartea | carte | NOUN | Case=Acc,Nom\|Definite=Def\|Gender=Fem\|Number=Sing | obj |
| sau | sau | CCONJ | Polarity=Pos | cc |
| să | să | PART | Mood=Sub | mark |
| o | el | PRON | Case=Acc\|Gender=Fem\|Number=Sing\|Person=3\|PronType=Prs\|Strength=Weak | obj |
| lase | lăsa | AUX | Mood=Sub\|Person=3\|Tense=Pres\|VerbForm=Fin | conj |
| ? | ? | PUNCT |  | punct |

