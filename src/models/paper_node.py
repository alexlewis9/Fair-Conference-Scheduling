from src.models.node import Node


class PaperNode(Node):
    """
    Represents a paper node with its metadata.
    """
    def __init__(self, node_id, emb, title, authors, year, abstract, publisher, session, pdf_url, openreview_url):
        """
        Initializes the paper node with its metadata.
        Args:
            node_id (int): Unique identifier for the node.
            emb (np.ndarray): Embedding (Coordinate) of the node.
            title (str): Title of the paper.
            authors (list): List of authors of the paper.
            year (int): Year of publication.
            abstract (str): Abstract of the paper.
            publisher (str): Publisher of the paper.
            session (str): Session in which the paper is presented.
            pdf_url (str): URL to the PDF of the paper.
            openreview_url (str): OpenReview URL of the paper.
        """
        super().__init__(node_id, emb)
        self.title = title
        self.authors = authors
        self.year = year
        self.abstract = abstract
        self.publisher = publisher
        self.session = session
        self.pdf_url = pdf_url
        self.openreview_url = openreview_url