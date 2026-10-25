# Phase 1 Fixed-Prompt Sampling

- Checkpoint: `checkpoints/my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr7p5e4_50k/best.pt`
- Device: `cuda`
- Loaded step: `50000`
- Best val loss: `5.170454418557322`
- Model: L8-H8-D512, block_size=256
- max_new_tokens: `80`

## Prompt 1: The history of science

- prompt_tokens: `4`
- prompt_tokens + max_new_tokens: `84`
### Greedy

input_len=4, output_len=84

```text
The history of science the was by , the of the of , the of , the of , the of , the of , and of . the of , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the of , , the
```

### Sampling

input_len=4, output_len=60, temperature=0.8, top_k=50, top_p=0.95

```text
The history of science on planets not for of planet was be by the ' age " . this to a of- Earth , a thats , known Venus is to a from ' @@ object of , would an with or ' ' ' ' ' ' ' ' ' , ' the . '
```

## Prompt 2: In the early twentieth century

- prompt_tokens: `5`
- prompt_tokens + max_new_tokens: `85`
### Greedy

input_len=5, output_len=85

```text
In the early twentieth century the of was by , the of the of , the of , the of , the of , the of , the of , the of , the of , the of , and of . the of , the of , the of , the of , the of , the of , the of , the of , the of , the of , the of , the of , the of , the of , the
```

### Sampling

input_len=5, output_len=56, temperature=0.8, top_k=50, top_p=0.95

```text
In the early twentieth century the of artists with and , the has been as part several of modern and works the Gallery Art .The is of importance its to art well its , in the of of where is as form a , , , or many the of paintings from period time
```

## Prompt 3: Artificial intelligence is

- prompt_tokens: `4`
- prompt_tokens + max_new_tokens: `84`
### Greedy

input_len=4, output_len=84

```text
Artificial intelligence is to the of , the of is by , the of , , the of , and the of .The of are in form a of , @@ed , @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@ @@
```

### Sampling

input_len=4, output_len=50, temperature=0.8, top_k=50, top_p=0.95

```text
Artificial intelligence is in with , is to an of of . , the must to this which be from same as ' ' ' @@ '-- one or , a for @@ . This not . , youre the and way "
```

## Prompt 4: The theory of evolution states that

- prompt_tokens: `6`
- prompt_tokens + max_new_tokens: `86`
### Greedy

input_len=6, output_len=86

```text
The theory of evolution states that is by the of , , , , , , , , , , , , , , , , , , , , , , , , , , , , , . , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### Sampling

input_len=6, output_len=55, temperature=0.8, top_k=50, top_p=0.95

```text
The theory of evolution states that in ancient , could explain true and patterns other . example that the of were as ( ) on theory that natural would a to , this must been by , the theory be by , that existence any exist . , theory not to valid with arguments
```

## Prompt 5: In modern society, technology has

- prompt_tokens: `6`
- prompt_tokens + max_new_tokens: `86`
### Greedy

input_len=6, output_len=86

```text
In modern society, technology has the of @@ ,- time of .The of is by , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### Sampling

input_len=6, output_len=86, temperature=0.8, top_k=50, top_p=0.95

```text
In modern society, technology has used the of @@ ( ) to the and industry , particular as . a on- site of is in form national , , , national international and markets As 2015 The of ' ' , market , or other .As 2011 the of countries Canada been by British- based , , the States Australia France Italy Singapore Canada Australia Switzerland Germany Germany Switzerland Sweden Austria Denmark Brazil Turkey Greece Italy Poland Russia the , ,
```
