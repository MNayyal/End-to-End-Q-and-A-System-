
# Build document store
corpus = [item["context"] for item in dataset["train"]]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_context(question, n=3):
    tokenized_question = question.split(" ")
    top_n = bm25.get_top_n(tokenized_question, corpus, n=n)
    return "\n\n".join(top_n)

def answer_question(question):
    # Retrieve relevant context
    context = retrieve_context(question)

    # Tokenize inputs
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt"
    )

    # Get prediction
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Decode answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end]
        )
    )

    return {
        "answer": answer,
        "context": context,
        "confidence": (
            torch.max(outputs.start_logits).item() +
            torch.max(outputs.end_logits).item()
        ) / 2
    }
