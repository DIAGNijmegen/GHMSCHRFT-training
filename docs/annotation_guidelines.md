# PHI Tagging Guidelines

These guidelines describe the different kinds of **Personal Health Information (PHI)** that can appear in reports and how to tag them.

**General rule:**  
If multiple consecutive PHI references relate to the same person or place, tag them as **one single tag**, not multiple.

---

## `<PERSOON>`

Includes:
- First names  
- Last names  
- Initials  

Tag any consecutive combination of these as **one tag**.

### Examples
- `Informatie doorgebeld aan radioloog A. Arends` → `Informatie doorgebeld aan radioloog <PERSOON>`
- `In behandeling bij Stephan` → `In behandeling bij <PERSOON>`
- `gebeld met collega Breedijk (hematologie)` → `gebeld met collega <PERSOON> (hematologie)`

### Notes
**Note I – Not PHI**  
Diseases or body parts with human names are **not PHI** and should not be tagged.

> Example:  
> `Patient bekend met Creutzfeldt-Jakob`

**Note II – Report-level references**  
Departments or specializations that apply to the entire report (and not to a specific person) should **not** be tagged.

> Example:  
> ```
> Aardmateriaal:
> Revisie prostaatbiopten
>
> Specialisme:
> Urologie
> ```

---

## `<PERSOONAFKORTING>`

Includes abbreviations referring to persons, including titles.

### Examples
- `Ingepakt door LPat` → `Ingepakt door <PERSOONAFKORTING>`
- `T.i. 1b. RRov/KZoe` → `T.i. 1b. <PERSOONAFKORTING>/<PERSOONAFKORTING>`
- `verslag geautoriseerd door AEij, KMBP` → `verslag geautoriseerd door <PERSOONAFKORTING>, KMBP`

---

## `<ZIEKENHUIS>`

Includes any reference to a hospital.  
Place names that implicitly refer to a hospital should also be tagged.

**Do not include departments.**

### Examples
- `Overgeplaatst naar Radboud Nijmegen` → `Overgeplaatst naar <ZIEKENHUIS>`
- `Verwijzing vanuit AMC` → `Verwijzing vanuit <ZIEKENHUIS>`
- `Doorverwezen vanuit Leiden zkhs` → `Doorverwezen vanuit <ZIEKENHUIS>`
- `Beelden uit Nijmegen` → `Beelden uit <ZIEKENHUIS>`
- `Genoomdiagnostiek Nijmegen` → `Genoomdiagnostiek <ZIEKENHUIS>`
- `Laboratorium Medische Microbiologie` → **not tagged**

---

## `<PLAATS>`

Includes all place names **not** referring to a hospital.

### Examples
- `Woont in Almere` → `Woont in <PLAATS>`
- `Persoon uit België` → `Persoon uit <PLAATS>`

---

## `<DATUM>`

Includes all date formats.

### Examples
- `Gemeten op 28/03` → `Gemeten op <DATUM>`
- `Laatste scan in 2011` → `Laatste scan in <DATUM>`
- `Vergeleken werd met scans van 13 en 29 juli` → `Vergeleken werd met scans van <DATUM> en <DATUM>`

---

## `<TIJD>`

Includes all time formats.

**Do not tag anatomical locations**, such as clock positions.

### Examples
- `Opname om 18:03` → `Opname om <TIJD>`
- `Scan genomen rond 20u` → `Scan genomen rond <TIJD>`
- `Laesie op 6 uur` → **not tagged**

---

## `<LEEFTIJD>`

Includes patient age.

### Examples
- `Patient is 49 jaar oud` → `Patient is <LEEFTIJD> jaar oud`
- `Leeftijd: 86 jaar` → `Leeftijd: <LEEFTIJD> jaar`

---

## `<TELEFOONNUMMER>`

Includes all internal and external phone numbers.

### Examples
- `Arts gebeld op sein 6162` → `Arts gebeld op <TELEFOONNUMMER>`
- `Doorgebeld op 45685` → `Doorgebeld op <TELEFOONNUMMER>`
- `Bereikbaar op 06-15468560` → `Bereikbaar op <TELEFOONNUMMER>`

---

## `<PATIENTNUMMER>`

Includes Radboudumc patient IDs (exactly 7 digits).

### Example
- `Patientnummer 1597557` → `Patientnummer <PATIENTNUMMER>`

---

## `<ZNUMMER>`

Includes Radboudumc personnel Z-numbers.

### Example
- `Geholpen door Z854913` → `Geholpen door <ZNUMMER>`

---

## `<RAPPORT_ID>`

Includes identification numbers for Radboudumc medical reports:
- T-numbers
- R-numbers
- C-numbers
- DPA-numbers
- RPA-numbers

### Examples
- `T12-38295` → `<RAPPORT_ID.T_NUMMER>`
- `R 04-167190` → `<RAPPORT_ID.R_NUMMER>`
- `C22-6116` → `<RAPPORT_ID.C_NUMMER>`
- `DPA18-00935_23849971` → `<RAPPORT_ID.DPA_NUMMER>`
- `RPA22-00473` → `<RAPPORT_ID.RPA_NUMMER>`

---

## `<PHINUMMER>`

Includes any PHI number not covered by:
`<PATIENTNUMMER>`, `<ZNUMMER>`, or `<RAPPORT_ID>`

Examples include external IDs, order numbers, etc.

### Examples
- `Patient 64852145` → `Patient <PHINUMMER>`
- `Zie order 160637831` → `Zie order <PHINUMMER>`

---

## `<ACCREDITATIE_NUMMER>`

Includes accreditation numbers of health institutes.

### Examples
- `onder ISO 15189 accreditatie M135` → `<ACCREDITATIE_NUMMER>`
- `Accreditatie nummer M142` → `<ACCREDITATIE_NUMMER>`

---

## `<STUDIE-NAAM>`

Includes specific study names and study codes  
(e.g. LEMA, PINNACLE, DONAN, MSPECT).

### Examples
- `LEMA studiecode UZR0291` → `<STUDIE-NAAM>`
- `LEMA studie` → `<STUDIE-NAAM>`
- `LEMA` → `<STUDIE-NAAM>`
- `donan 108329` → `<STUDIE-NAAM>`

---

## `<URL>`

Includes all URLs.

### Example
- `https://www.palga.nl/datasheet/Radboudumc/TSO500.pdf` → `<URL>`

---

## `<BEDRIJF>`

Includes organization names **only when they indicate the employer or physical workplace of a specific person**.

**Do not use for:**
- Manufacturers  
- Vendors  
- Suppliers  
- Test kit producers  
- Organizations not linked to a specific person  

### Examples
- `Werkzaam bij Sanquin` → `Werkzaam bij <BEDRIJF>`
- `Patholoog in dienst van Roche` → `Patholoog in dienst van <BEDRIJF>`

---

## `<EMAIL>`

Includes all internal and external email addresses.

### Examples
- `Contact via jansen@radboudumc.nl` → `Contact via <EMAIL>`
- `Mail gestuurd naar info@labservice.com` → `Mail gestuurd naar <EMAIL>`

---

## `<IBAN>`

Includes all IBAN numbers, regardless of country.

### Examples
- `IBAN: NL91ABNA0417164300` → `IBAN: <IBAN>`
- `Rekeningnummer BE68 5390 0754 7034` → `Rekeningnummer <IBAN>`

---

## `<BSN>`

Includes all Burgerservicenummers (BSN).

**Always tag BSNs as `<BSN>`, never as `<PHINUMMER>`.**

### Examples
- `BSN 123456782` → `<BSN>`
- `BSN: 228956781` → `<BSN>`

---

## `<BIGNUMMER>`

Includes BIG registration numbers of healthcare professionals.

### Examples
- `BIG-nummer 19012345678` → `<BIGNUMMER>`
- `BIG registratie: 29098765432` → `<BIGNUMMER>`

---

## `<AGBNUMMER>`

Includes AGB codes of healthcare providers or institutions.

### Examples
- `AGB-code 03001234` → `<AGBNUMMER>`
- `AGB nummer: 08009999` → `<AGBNUMMER>`

## `<ADRES>`

Includes all full addresses, including street names, house numbers, postal codes, and city names.

### Examples
- `Woonadres: Kerkstraat 12, 1234 AB Amsterdam` → `Woonadres: <ADRES>`

```Adres:
Kerkstraat 12
1234 AB Amsterdam
```
→
```Adres:
<ADRES>
``` 