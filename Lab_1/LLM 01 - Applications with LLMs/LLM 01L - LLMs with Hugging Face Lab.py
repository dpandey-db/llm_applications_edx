# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LLMs with Hugging Face
# MAGIC
# MAGIC **Choosing a pre-trained LLM**: In the demo notebook, you saw how to apply pre-trained models to many applications.  You will do that hands-on in this lab, with your main activity being to find a good model for each task.  Use the tips from the lecture and demo to find good models, and don't hesitate to try a few different possibilities.
# MAGIC
# MAGIC **Understanding LLM pipeline configurations**: At the end of this lab, you will also do a more open-ended exploration of model and tokenizer configurations.
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Practice finding an existing model for tasks you want to solve with LLMs.
# MAGIC 1. Understand the basics of model and tokenizer options for tweaking model outputs and performance.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Find good models for your tasks
# MAGIC
# MAGIC In each subsection below, you will solve a given task with an LLM of your choosing.  These tasks vary from straightforward to open-ended:
# MAGIC * **Summarization**: There are many summarization models out there, and many are simply plug-and-play.
# MAGIC * **Translation**: This task can require more work since models support varying numbers of languages, and in different ways.  Make sure you invoke your chosen model with the right parameters.
# MAGIC * **Few-shot learning**: This task is very open-ended, where you hope to demonstrate your goals to the LLM with just a few examples.  Choosing those examples and phrasing your task correctly can be more art than science.
# MAGIC
# MAGIC Recall these tips from the lecture and demo:
# MAGIC * Use the [Hugging Face Hub](https://huggingface.co/models).
# MAGIC * Filter by task, license, language, etc. as needed.
# MAGIC * If you have limited compute resources, check model sizes to keep execution times lower.
# MAGIC * Search for existing examples as well.  It can be helpful to see exactly how models should be loaded and used.

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline

# COMMAND ----------

# MAGIC %md ### Question 1: Summarization
# MAGIC
# MAGIC In this section, you will find a model from the Hugging Face Hub for a new summarization problem. **Do not use a T5 model**; find and use a model different from the one we used in the demo notebook.
# MAGIC
# MAGIC We will use the same [xsum](https://huggingface.co/datasets/xsum) dataset.

# COMMAND ----------

xsum_dataset = load_dataset(
    "xsum", version="1.2.0", cache_dir=DA.paths.datasets
)  # Note: We specify cache_dir to use predownloaded data.
xsum_sample = xsum_dataset["train"].select(range(10))
display(xsum_sample.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly to how we found and applied a model for summarization previously, fill in the missing parts below to create a pipeline using an existing LLM---but with a different model.  Then apply the pipeline to the sample batch of articles.

# COMMAND ----------

# TODO

# Constructor a summarization pipeline
summarizer = pipeline(
  task="summarization",
  model="google/pegasus-newsroom",
  min_length=20,
  max_length=40,
  truncation=True,
  model_kwargs={"cache_dir": DA.paths.datasets},
)

# Apply the pipeline to the batch of articles in `xsum_sample`
summarization_results = summarizer(xsum_sample['document'])

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion1_1(summarizer, summarization_results, xsum_sample["document"])

# COMMAND ----------

# Display the generated summary side-by-side with the reference summary and original document.
import pandas as pd

display(
    pd.DataFrame.from_dict(summarization_results)
    .rename({"summary_text": "generated_summary"}, axis=1)
    .join(pd.DataFrame.from_dict(xsum_sample))[
        ["generated_summary", "summary", "document"]
    ]
)

# COMMAND ----------

# MAGIC %md ### Question 2: Translation
# MAGIC
# MAGIC In this section, you will find a model from the Hugging Face Hub for a new translation problem.
# MAGIC
# MAGIC We will use the [Helsinki-NLP/tatoeba_mt](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) dataset.  It includes sentence pairs from many languages, but we will focus on translating Japanese to English.
# MAGIC
# MAGIC If you feel stuck, please refer to the hints at the end of this section.

# COMMAND ----------

jpn_dataset = load_dataset(
    "Helsinki-NLP/tatoeba_mt",
    version="1.0.0",
    language_pair="eng-jpn",
    cache_dir=DA.paths.datasets,
)
jpn_sample = (
    jpn_dataset["test"]
    .select(range(10))
    .rename_column("sourceString", "English")
    .rename_column("targetString", "Japanese")
    .remove_columns(["sourceLang", "targetlang"])
)
display(jpn_sample.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Similarly to how we previously found and applied a model for translation among other languages, you must now find a model to translate from Japanese to English.  Fill in the missing parts below to create a pipeline using an existing LLM.  Then apply the pipeline to the sample batch of Japanese sentences.

# COMMAND ----------

# TODO

# Construct a pipeline for translating Japanese to English.
translation_pipeline = pipeline(
  task='translation',
  model='staka/fugumt-ja-en',
  model_kwargs={"cache_dir": DA.paths.datasets},
)

# Apply your pipeline on the sample of Japanese text in: jpn_sample["Japanese"]
translation_results = translation_pipeline(jpn_sample["Japanese"])

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion1_2(translation_pipeline, translation_results, jpn_sample["Japanese"])

# COMMAND ----------

# Now we can display your translations side-by-side with the ground-truth `English` column from the dataset.
translation_results_df = pd.DataFrame.from_dict(translation_results).join(
    jpn_sample.to_pandas()
)
display(translation_results_df)

# COMMAND ----------

# MAGIC %md If you feel stuck on the above Japanese -> English translation task, here are some hints:
# MAGIC * Some models can handle *a lot* of languages.  Check out [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb), the No Language Left Behind model ([research paper](https://arxiv.org/abs/2207.04672)).
# MAGIC * The "translation" task for `pipeline` takes optional parameters `src_lang` (source language) and `tgt_lang` (target language), which are important when the model can handle multiple languages.  To figure out what codes to use to specify languages (and scripts for those languages), it can be helpful to find existing examples of using your model; for NLLB, check out [this Python script with codes](https://huggingface.co/spaces/Geonmo/nllb-translation-demo/blob/main/flores200_codes.py) or similar demo resources.
# MAGIC

# COMMAND ----------

# MAGIC %md ### Question 3: Few-shot learning
# MAGIC
# MAGIC In this section, you will build a prompt which gets an LLM to answer a few-shot learning problem.  Your prompt will have 3 sections:
# MAGIC
# MAGIC 1. High-level instruction about the task
# MAGIC 1. Examples of query-answer pairs for the LLM to learn from
# MAGIC 1. New query
# MAGIC
# MAGIC Your goal is to make the LLM answer the new query, with as good a response as possible.
# MAGIC
# MAGIC More specifically, your prompt should following this template:
# MAGIC ```
# MAGIC <High-level instruction about the task: Given input_label, generate output_label.>:
# MAGIC
# MAGIC [<input_label>]: "<input text>"
# MAGIC [<output_label>]: "<output_text>"
# MAGIC ###
# MAGIC [<input_label>]: "<input text>"
# MAGIC [<output_label>]: "<output_text>"
# MAGIC ###
# MAGIC [<input_label>]: "<input text>"
# MAGIC [<output_label>]:
# MAGIC ```
# MAGIC where the final two lines represent the new query.
# MAGIC
# MAGIC It is up to you to choose a task, but here are some ideas:
# MAGIC * Translation: This is easy but less interesting since there are already models fine-tuned for translation.  You can generate examples via a tool like Google Translate.
# MAGIC * Create book titles or descriptions: Given a book title, generate a description, or vice versa.  You can get examples off of Wikipedia.
# MAGIC * Generate tweets: Given keywords or a key message, generate a tweet.
# MAGIC * Identify the subject: Given a sentence, extract the noun or name of the subject of the sentence.
# MAGIC
# MAGIC *Please **do not** copy examples from the demo notebook.*
# MAGIC
# MAGIC Tips:
# MAGIC * If the model gives bad outputs with only 1 or 2 examples, try adding more.  3 or 4 examples can be much better than 1 or 2.
# MAGIC * Not all tasks are equally difficult.  If your task is too challenging, try a different one.

# COMMAND ----------

few_shot_pipeline = pipeline(
    task="text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    max_new_tokens=50,
    model_kwargs={"cache_dir": DA.paths.datasets},
)  # Use a predownloaded model

# Get the token ID for "###", which we will use as the EOS token below.  (Recall we did this in the demo notebook.)
eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Fill in the template below.  Feel free to adjust the number of examples.

# COMMAND ----------

# TODO

# Fill in this template.

prompt =\
"""Given the back cover blurb from a book, pick the most likely genre:

[Blurb]: "Harry Potter has no idea how famous he is. That's because he's being raised by his miserable aunt and uncle who are terrified Harry will learn that he's really a wizard, just as his parents were. But everything changes when Harry is summoned to attend an infamous school for wizards, and he begins to discover some clues about his illustrious birthright. From the surprising way he is greeted by a lovable giant, to the unique curriculum and colorful faculty at his unusual school, Harry finds himself drawn deep inside a mystical world he never knew existed and closer to his own noble destiny."
[Genre]: "Fantasy"
###
[Blurb]: "At the dawn of the nineteenth century, two very different magicians emerge to change England's history. In the year 1806, with the Napoleonic Wars raging on land and sea, most people believe magic to be long dead in England--until the reclusive Mr. Norrell reveals his powers, and becomes a celebrity overnight. Yet the cautious, fussy Norrell is challenged by the emergence of another magician, the brilliant novice Jonathan Strange. Young, handsome, and daring, Strange is the very opposite of Norrell. He becomes Norrell’s student, and they join forces in the war against France. But Strange is increasingly drawn to the wildest, most perilous forms of magic, straining his partnership with Norrell, and putting at risk everything else he holds dear."
[Genre]: "Fantasy"
###
[Blurb]: "Hiro lives in a Los Angeles where franchises line the freeway as far as the eye can see. The only relief from the sea of logos is within the autonomous city-states, where law-abiding citizens don’t dare leave their mansions. Hiro delivers pizza to the mansions for a living, defending his pies from marauders when necessary with a matched set of samurai swords. His home is a shared 20 X 30 U-Stor-It. He spends most of his time goggled in to the Metaverse, where his avatar is legendary. But in the club known as The Black Sun, his fellow hackers are being felled by a weird new drug called Snow Crash that reduces them to nothing more than a jittering cloud of bad digital karma (and IRL, a vegetative state). Investigating the Infocalypse leads Hiro all the way back to the beginning of language itself, with roots in an ancient Sumerian priesthood. He’ll be joined by Y.T., a fearless teenaged skateboard courier. Together, they must race to stop a shadowy virtual villain hell-bent on world domination."
[Genre]: "Science Fiction"
###
[Blurb]: "Attorney Billy Halleck seriously enjoys living his life of upper-class excess. He’s got it all­—an expensive home in Connecticut, a loving family…and fifty extra pounds that his doctor repeatedly warns will be the death of him. Then, in a moment of carelessness, Halleck commits vehicular manslaughter when he strikes a jaywalking old woman crossing the street. But Halleck has some powerful local connections, and gets off with a slap on the wrist…much to the fury of the woman’s mysterious and ancient father, who exacts revenge with a single word: “Thinner.” Now a terrified Halleck finds the weight once so difficult to shed dropping effortlessly—and rapidly—by the week. Soon there will be nothing left of Billy Halleck…unless he can somehow locate the source of his living nightmare and reverse what’s happened to him before he utterly wastes away…"
[Genre]:"""

# COMMAND ----------

results = few_shot_pipeline(prompt, do_sample=True, eos_token_id=eos_token_id)

print(results[0]["generated_text"])

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion1_3(few_shot_pipeline, prompt, results[0]["generated_text"])

# COMMAND ----------

# MAGIC %md ## Explore model and tokenizer settings
# MAGIC
# MAGIC So far, we have used pipelines in a very basic way, without worrying about configuration options.  In this section, you will explore the various options for models and tokenizers to learn how they affect LLM behavior.
# MAGIC
# MAGIC We will load a dataset, tokenizer, and model for you.  We will also define a helper method for printing out results nicely.

# COMMAND ----------

# Load data, tokenizer, and model.

from transformers import T5Tokenizer, T5ForConditionalGeneration

xsum_dataset = load_dataset("xsum", version="1.2.0", cache_dir=DA.paths.datasets)
xsum_sample = xsum_dataset["train"].select(range(10))

tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=DA.paths.datasets)
model = T5ForConditionalGeneration.from_pretrained(
    "t5-small", cache_dir=DA.paths.datasets
)

# Prepare articles for T5, which requires a "summarize: " prefix.
articles = list(map(lambda article: "summarize: " + article, xsum_sample["document"]))

# COMMAND ----------

def display_summaries(decoded_summaries: list) -> None:
    """Helper method to display ground-truth and generated summaries side-by-side"""
    results_df = pd.DataFrame(zip(xsum_sample["summary"], decoded_summaries))
    results_df.columns = ["Summary", "Generated"]
    display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Open-ended exploration
# MAGIC
# MAGIC In the cells below, we provide code for running the tokenizer and model on the articles.  Your task is to play around with the various configurations to gain more intuition about the effects.  Look for changes to output quality and running time in particular, and remember that running the same code twice may result in different answers.
# MAGIC
# MAGIC Below, we list brief descriptions of each of the parameters you may wish to tweak.
# MAGIC * Tokenizer encoding
# MAGIC   * `max_length`: This caps the maximum input length.  It must be at or below the model's input length limit.
# MAGIC   * `return_tensors`: Do not change this one.  This tells Hugging Face to return tensors in PyTorch ("pt") format.
# MAGIC * Model
# MAGIC   * `do_sample`: True or False.  This tells the model whether or not to use sampling in generation.  If False, then it will do greedy search or beam search.  If True, then it will do random sampling which can optionally use the top-p and/or top-k sampling techniques.  See the blog post linked below for more details on sampling techniques.
# MAGIC   * `num_beams`: (for beam search) This specifies the number of beams to use in beam search across possible sequences.  Increasing the number can help the model to find better sequences, at the cost of more computation.
# MAGIC   * `min_length`, `max_length`: Generative models can be instructed to generate new text between these token lengths.
# MAGIC   * `top_k`: (for sampling) This controls the use of top-K sampling, which forces sampling to ignore low-probability tokens by limiting to the K most probable next tokens.  Set to 0 to disable top-K sampling.
# MAGIC   * `top_p`: (for sampling) This controls the use of top-p sampling, which forces sampling to ignore low-probability tokens by limiting to the top tokens making up probability mass p.  Set to 0 to disable top-p sampling.
# MAGIC   * `temperature`: (for sampling) This controls the "temperature" of the softmax.  Lower values bias further towards high-probability next tokens.  Setting to 0 makes sampling equivalent to greedy search.
# MAGIC * Tokenizer decoding
# MAGIC   * `skip_special_tokens`: True or False.  This allows you to skip special tokens (like EOS tokens) in the model outputs.
# MAGIC
# MAGIC Do not tweak:
# MAGIC * Tokenizer encoding
# MAGIC   * `padding`: True or False.  This helps to handle variable-length inputs by adding padding to short inputs.  Since it should be set according to your task and data, you should not change it for this exercise (unless you want to see what warnings or error may appear).
# MAGIC   * `truncation`: True or False.  This helps to handle variable-length inputs by truncating very long inputs.  Since it should be set according to your task and data, you should not change it for this exercise (unless you want to see what warnings or error may appear).
# MAGIC
# MAGIC If you need more info about the parameters of methods, see the `help()` calls in cells below, or search the Hugging Face docs.  Some top links are:
# MAGIC * Tokenizer call for encoding: [PreTrainedTokenizerBase.\_\_call\_\_ API docs](https://huggingface.co/docs/transformers/v4.28.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__)
# MAGIC * Model invocation: [Docs for generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) and this blog post on ["How to generate text: using different decoding methods for language generation with Transformers"](https://huggingface.co/blog/how-to-generate)
# MAGIC
# MAGIC If you mess up and can't get back to a working state, you can use the Revision History to revert your changes.
# MAGIC Access that via the clock-like icon or "Revision History" button in the top-right of this notebook page. (See screenshot below.)
# MAGIC
# MAGIC ![Screenshot of notebook Revision History](https://files.training.databricks.com/images/llm/revision_history.png)

# COMMAND ----------

# DBTITLE 1,Default provided cell
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=2,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.5,
    temperature=0.7,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# DBTITLE 1,skip_special_tokens = false
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=2,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.5,
    temperature=0.7,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=False)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# DBTITLE 1,Halving max length
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=512, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=2,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.5,
    temperature=0.7,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# DBTITLE 1,Switch values for top_p and temperature
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=2,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.7,
    temperature=0.5,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# DBTITLE 1,Increase top_p to cover a larger AUC
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=2,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.7,
    temperature=0.7,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# DBTITLE 1,Can increasing num_beams help with the weird summaries we got by increasing top_p?
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=3,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.7,
    temperature=0.7,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# DBTITLE 1,Increasing num_beams on the given default cell
##############################################################################
# TODO: Try editing the parameters in this section, and see how they affect the results.
#       You can also copy and edit the cell to compare results across different parameter settings.
#
# We show all parameter settings for ease-of-modification, but in practice, you would only set relevant ones.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)

summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    do_sample=True,
    num_beams=3,
    min_length=0,
    max_length=40,
    top_k=20,
    top_p=0.5,
    temperature=0.7,
)

decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
##############################################################################

display_summaries(decoded_summaries)

# COMMAND ----------

# MAGIC %md Uncomment `help()` calls below as needed to see doc strings for stages of the pipeline.

# COMMAND ----------

# Options for calling the tokenizer (lots to see here)
help(tokenizer.__call__)

# COMMAND ----------

# Options for invoking the model (lots to see here)
help(model.generate)

# COMMAND ----------

# Options for calling the tokenizer for decoding (not much to see here)
help(tokenizer.batch_decode)

# COMMAND ----------

# MAGIC %md ## Submit your Results (edX Verified Only)
# MAGIC
# MAGIC To get credit for this lab, click the submit button in the top right to report the results. If you run into any issues, click `Run` -> `Clear state and run all`, and make sure all tests have passed before re-submitting. If you accidentally deleted any tests, take a look at the notebook's version history to recover them or reload the notebooks.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
