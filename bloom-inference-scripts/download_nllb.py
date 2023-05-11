from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("------- loading tokenizer -------")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")

print("------- loading model -------")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-moe-54b")
print("------- successfuly load model -------")

batched_input = ['We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.',
"Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days."
"Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes."
"On Monday, Sara Danius, permanent secretary of the Nobel Committee for Literature at the Swedish Academy, publicly announced during a radio program on Sveriges Radio in Sweden the committee, unable to reach Bob Dylan directly about winning the 2016 Nobel Prize in Literature, had abandoned its efforts to reach him.",
'Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough."',
"Previously, Ring's CEO, Jamie Siminoff, remarked the company started when his doorbell wasn't audible from his shop in his garage.",
]
inputs = tokenizer(batched_input, return_tensors="pt", padding = True)

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"]
)
tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

print("--- exit ---")