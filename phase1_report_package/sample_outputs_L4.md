# Phase 1 Fixed-Prompt Sampling

- Checkpoint: `checkpoints/my_adamw_L4_H4_D256_bs8_blk256_ts256_/best.pt`
- Device: `cuda`
- Loaded step: `50000`
- Best val loss: `5.55388731834216`
- Model: L4-H4-D256, block_size=256
- max_new_tokens: `80`

## Prompt 1: The history of science

- prompt_tokens: `4`
- prompt_tokens + max_new_tokens: `84`
### Greedy

input_len=4, output_len=84

```text
The history of science- fiction the was by and by , the of ' " @@ " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "
```

### Sampling

input_len=4, output_len=84, temperature=0.8, top_k=50, top_p=0.95

```text
The history of science was in ; is and by , is by to that the were @@ of- fiction , published 18 , by British magazine Publishing and & . novel in novel novels and is as by R B DJ Abrams with M and , with H ' being on a from of ' ' ' name K ' ' original , ' ' 's . film published 18 18 , , The and works been for years
```

## Prompt 2: In the early twentieth century

- prompt_tokens: `5`
- prompt_tokens + max_new_tokens: `85`
### Greedy

input_len=5, output_len=85

```text
In the early twentieth century the of , was by and in , the of was to a of @@ period the of . the of was by , the of was by of , the of , the of was by of , the of and of , the of were by of , the of and of were by . the of was by of , the of was by of , the of was by of and by of , the
```

### Sampling

input_len=5, output_len=85, temperature=0.8, top_k=50, top_p=0.95

```text
In the early twentieth century the of the andteenth , French of (..th ) used to a @@ range trade a from @@ to areas such China China Japan which to the Kingdom an population about % the , by thes were in and countries as . , and population the government an group refugees India mostly their , , China Spain Portugal other , and , well their and . @@ indigenous are from Chinese such
```

## Prompt 3: Artificial intelligence is

- prompt_tokens: `4`
- prompt_tokens + max_new_tokens: `84`
### Greedy

input_len=4, output_len=84

```text
Artificial intelligence is to the of , the of and of , the of are by .The of is in with , the of is by of , the of , the of , the of , the of , the of , the of , the of , the of , the , the of , the , the of , the , the of , the , the , , the of , the , the , the , ,
```

### Sampling

input_len=4, output_len=29, temperature=0.8, top_k=50, top_p=0.95

```text
Artificial intelligence is used the to a , and of , can be by , in they not the for purpose that work an function this .
```

## Prompt 4: The theory of evolution states that

- prompt_tokens: `6`
- prompt_tokens + max_new_tokens: `86`
### Greedy

input_len=6, output_len=86

```text
The theory of evolution states that is to the of , the of and of , the of is by a of , , , , the of , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### Sampling

input_len=6, output_len=81, temperature=0.8, top_k=50, top_p=0.95

```text
The theory of evolution states that evolutionary has been to in form evolution and , in analysis the @@ theories evolution as and evolution evolution that genetic was and . the of ' is most of early evolution thes a world In ( ) was by evolutionary , the of @@ and @@ groups that not is the of .In evolution this is to to that otherism more an are closely in or selection
```

## Prompt 5: In modern society, technology has

- prompt_tokens: `6`
- prompt_tokens + max_new_tokens: `86`
### Greedy

input_len=6, output_len=86

```text
In modern society, technology has to the of and , is by a of @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @
```

### Sampling

input_len=6, output_len=56, temperature=0.8, top_k=50, top_p=0.95

```text
In modern society, technology has been in of and , its is to the that is on ones of best .According this , of importance and the of class womens a are by . the of classes there no of in schools as as primary ( ) their are of students
```
