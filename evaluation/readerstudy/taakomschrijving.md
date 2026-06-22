# Taakomschrijving — Lezersstudie anonimisering

## Achtergrond

In het kader van onderzoek naar automatische anonimisering van medische verslagen
beoordelen wij de kwaliteit van een anonimiseringsmodel. De pipeline werkt in twee stappen:

1. **Detectie** — een model detecteert persoonsidentificerende informatie (PII) in het verslag.
2. **HIPS** (*Hiding In Plain Sight*) — gedetecteerde PII wordt vervangen door realistische vervangende waarden (bijv. een andere naam, een andere datum). Het verslag blijft daardoor leesbaar.

Het resultaat is een verslag waarin de gevonden PII is vervangen, maar waarin eventueel gemiste PII nog steeds als originele tekst aanwezig is.

---

## Wat je beoordeelt

Je krijgt verslagen te zien **na** verwerking door de pipeline. Sommige verslagen zijn volledig geanonimiseerd. In andere verslagen heeft het model PII gemist, waardoor er nog originele identificerende informatie in de tekst staat.

**Belangrijk:** je taak is uitsluitend het opsporen van PII die het model heeft **gemist** — dat wil zeggen, originele tekst die identificerend is en nog niet vervangen is. Je hoeft je geen zorgen te maken over de vervangen waarden zelf. Dat die er staan is correct. In sommige verslagen zal het duidelijk zijn dat er vervangingen hebben plaatsgevonden; dat is geen fout.

---

## Hoe je annoteert

Gebruik **twee labels** om verdachte tekststukken te markeren:

| Label | Gebruik |
|---|---|
| **ZEKER_FOUT** | Dit is duidelijk originele PII die niet vervangen is — je herkent het direct als identificerend (bijv. een echte naam, een echt patiëntnummer) |
| **MOGELIJK_FOUT** | Dit ziet er verdacht uit en zou identificerend kunnen zijn, maar je bent er niet zeker van — markeer het voor zekerheid |

Markeer het exacte stukje tekst dat identificerend is. Als er meerdere fouten in één verslag staan, markeer ze dan allemaal.

Als je niets verdachts ziet, laat je de annotatie leeg en ga je door naar het
volgende verslag.

---

## Soorten PII

De volgende typen persoonsidentificerende informatie zijn relevant:

| Categorie | Voorbeelden |
|---|---|
| Persoonsnamen | volledige namen, initialen, afkortingen van zorgverleners of patiënten |
| Datums en tijden | geboortedata, onderzoeksdatums, tijdstippen |
| Leeftijd | leeftijd in jaren of maanden |
| Telefoonnummers | alle telefoonnummers |
| Adressen en plaatsen | straatnamen, postcodes, plaatsnamen |
| Ziekenhuizen en instellingen | namen van ziekenhuizen of klinieken |
| Identificatienummers | patiëntnummers, rapportnummers, BSN, BIG-nummers, AGB-nummers, accreditatienummers |
| Overig | e-mailadressen, URL's, IBAN, studienamen, overige identificerende informatie |

---

## Vragen of onduidelijkheden

Neem contact op met <luc.builtjes@radboudumc.nl>.
