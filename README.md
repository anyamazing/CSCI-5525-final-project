# CSCI-5525-final-project

Running this is a bit of a pain, might add a stream-lined way of running things since my local is a little bit cursed.

The Jupyter Notebook `Process_with_augmentations.ipynb` is a great start that does not require going through pain and suffering to set up  quantized transformers.

to set up quantized transformers:

```shell
git clone https://github.com/anyamazing/CSCI-5525-final-project
cd CSCI-5525-final-project
git clone https://github.com/stangelid/qt/
```

To set up QT, you can train following the [default instructions](https://github.com/stangelid/qt#training-qt) or go through [custom corpus training](https://github.com/stangelid/qt/blob/main/custom.md) though they work the same and we have no numerical results for the end-to-end so might as well suffer less. If you want to go through the custom process, generate data with the `generate_data.py` script and move the files into `qt/data/json`. The fashion JSON is not large enough, I would recommend using Kitchen and Appliances though it takes about 30 GB of RAM. Alternatively, the vanilla training comes with all the data needed and takes just over 8 GB of VRAM.

Once all is said and done, from `qt/src` you will be able to run

```shell
# write general summaries
python extract.py --summary_data ../data/json/music_summ.json --no_eval --model ../models/model.pt --sample_sentences --split presplit --no_early_stop --no_cut_sents --newline_sentence_split --max_tokens 300 --run_id music

# write aspect (topic) specific summaries (sorry about all the aspect jank)
python aspect_extract.py --summary_data ../data/json/music_summ.json --no_eval --model ../models/model.pt --seedsdir ../data/myseeds --gold_aspects Gifts,Guitars,Jobs,Music,Services,Shipping,Sizes,Strings,Tools --sample_sentences --split presplit --no_early_stop --no_cut_sents --newline_sentence_split --max_tokens 300 --run_id musicasp
```
