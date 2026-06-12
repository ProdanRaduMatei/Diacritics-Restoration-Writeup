# Manual Hard Cases

These examples are intentionally not cherry-picked successes. They test where a frequency-only system and an under-trained contextual model struggle: meaning, definiteness, agreement, clitics and punctuation-adjacent words. The `constraint` column matters because the final pipeline should stay safe even when the chosen diacritized form is wrong.

| source | expected | prediction | layer | constraint | why hard |
|---|---|---|---|---:|---|
| `Am pus peste peste legume.` | `Am pus pește peste legume.` | `Am pus peste peste legume.` | transformer | True | The first `peste` is a noun and the second is a preposition; corpus frequency alone prefers `peste`. |
| `Fata sta in fata casei.` | `Fata stă în fața casei.` | `Fața sta în fața casei.` | dictionary_fallback | True | `fata` can be `fata`, `fată`, `fața`; the second position is forced by the prepositional phrase. |
| `Mana dreapta era in mana medicului.` | `Mâna dreaptă era în mâna medicului.` | `Mâna dreapta era în mâna medicului.` | dictionary_fallback | True | `mana` differs between indefinite `mână` and definite `mâna`; local syntax matters. |
| `Para dulce nu e acelasi lucru cu o para.` | `Para dulce nu e același lucru cu o pară.` | `Pară dulce nu e același lucru cu o pară.` | dictionary_fallback | True | `para` may stay plain in named/foreign contexts, but fruit needs `pară`. |
| `Casa veche nu era acasa.` | `Casa veche nu era acasă.` | `Casă veche nu era acasă.` | dictionary_fallback | True | `casa/casă` and `acasă` show that token frequency is not enough; phrase role is important. |
| `Sa-si ia cartea sau sa o lase?` | `Să-și ia cartea sau să o lase?` | `Sa-și ia cartea sau să o lase?` | transformer | True | Hyphenated clitics test whether the system handles punctuation-adjacent words. |
| `Cand vin peste o vreme acasa, casa nu se va simti ca acasa, ci cu desavarsire va fi o alta fatada a unei cladiri.` | `Când vin peste o vreme acasă, casa nu se va simți ca acasă, ci cu desăvârșire va fi o altă fațadă a unei clădiri.` | `Când vin peste o vreme acasă, casă nu se va simți că acasă, ci cu desăvârșire va fi o altă fațada a unei cladiri.` | dictionary_fallback | True | Multiple ambiguous tokens interact: `casa/casă`, `ca/că`, `fațada/fațadă`; the output is safe but semantically wrong in several places. |
