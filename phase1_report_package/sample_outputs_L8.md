# Phase 1 Fixed-Prompt Sampling

- Checkpoint: `checkpoints/my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel/best.pt`
- Device: `cuda`
- Loaded step: `50000`
- Best val loss: `5.334591590441191`
- Model: L8-H8-D512, block_size=256
- max_new_tokens: `80`

## Prompt 1: The history of science

- prompt_tokens: `4`
- prompt_tokens + max_new_tokens: `84`
### Greedy

input_len=4, output_len=60

```text
The history of science the was by , the of was to a of @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ .
```

### Sampling

input_len=4, output_len=84, temperature=0.8, top_k=50, top_p=0.95

```text
The history of science and literature both and in science it been by historians the of writers thes The and . scholars a by- authors have that book should seen the as @@ , , , @@ , , , @@ , , , , and be with or .The has been to an impact some of world since earlyth and centuries and been by artists as in . was and in cases the is as ,
```

## Prompt 2: In the early twentieth century

- prompt_tokens: `5`
- prompt_tokens + max_new_tokens: `85`
### Greedy

input_len=5, output_len=85

```text
In the early twentieth century the of was as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as . , the of and the of , as as
```

### Sampling

input_len=5, output_len=37, temperature=0.8, top_k=50, top_p=0.95

```text
In the early twentieth century it an in form classical , style music a of and music In with from beginning thes was , all . of original styles been with classical as of style
```

## Prompt 3: Artificial intelligence is

- prompt_tokens: `4`
- prompt_tokens + max_new_tokens: `84`
### Greedy

input_len=4, output_len=84

```text
Artificial intelligence is to the of , the of , and the of .The of are by the of , the of , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### Sampling

input_len=4, output_len=22, temperature=0.8, top_k=50, top_p=0.95

```text
Artificial intelligence is to a , or . , ' is the of who that are to . ' '
```

## Prompt 4: The theory of evolution states that

- prompt_tokens: `6`
- prompt_tokens + max_new_tokens: `86`
### Greedy

input_len=6, output_len=86

```text
The theory of evolution states that is to the of and the of , the of is by the of . , is in case the of , it be as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as as
```

### Sampling

input_len=6, output_len=86, temperature=0.8, top_k=50, top_p=0.95

```text
The theory of evolution states that are and , with , the of , in , , the has on . , can be by from . most thees the and is to to ones the of ; example if may be more less than of (.. in case an ) Forism which not related is .However thein is as the of of non- exist , its ( ) have been in same as (.. )
```

## Prompt 5: In modern society, technology has

- prompt_tokens: `6`
- prompt_tokens + max_new_tokens: `86`
### Greedy

input_len=6, output_len=86

```text
In modern society, technology has the of and , is to a extent the of .The of is in form a of , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### Sampling

input_len=6, output_len=37, temperature=0.8, top_k=50, top_p=0.95

```text
In modern society, technology has more a of @@ jobs the rate time to , is used a in- market . , is used promote in and , as result a shortage money
```
