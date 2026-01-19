import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_text_short():
    return "This is a short sentence. This is another sentence."


@pytest.fixture
def sample_text_medium():
    return (
        "Natural language processing is a subfield of linguistics and artificial intelligence. "
        "It focuses on the interaction between computers and human language. "
        "The goal is to enable computers to understand, interpret, and generate human language. "
        "This involves various tasks such as text classification, sentiment analysis, and machine translation."
    )


@pytest.fixture
def sample_text_long():
    return (
        "Machine learning is a branch of artificial intelligence that focuses on building systems "
        "that can learn from data. These systems improve their performance over time without being "
        "explicitly programmed. The field has seen tremendous growth in recent years, driven by "
        "advances in computing power and the availability of large datasets. "
        "Deep learning, a subset of machine learning, has been particularly successful. "
        "It uses artificial neural networks with multiple layers to model complex patterns in data. "
        "Applications of machine learning span various domains including computer vision, natural "
        "language processing, robotics, and healthcare. Neural networks, inspired by biological "
        "neurons, form the foundation of many modern AI systems. These networks can learn hierarchical "
        "representations of data, making them powerful tools for solving complex problems. "
        "The future of AI and machine learning holds immense potential for transforming industries "
        "and improving our daily lives."
    )


@pytest.fixture
def sample_documents():
    return [
        "First document with some content. It has multiple sentences.",
        "Second document is here. It also contains several sentences for testing purposes.",
        "Third and final document. This one is shorter."
    ]