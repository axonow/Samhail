
"""
MarkovChain Text Generator

This module implements a simple Markov Chain-based text generator. It allows training on a given text corpus
and generating new text sequences based on the learned Markov Chain model.

Classes:
    - MarkovChain: Represents the Markov Chain model for text generation.

Methods:
    - __init__: Initializes the Markov Chain with an empty graph.
    - _tokenize: Tokenizes input text by removing punctuation, numbers, and splitting into words.
    - train: Trains the Markov Chain by building a graph of word transitions from the input text.
    - generate: Generates a sequence of text based on the trained Markov Chain and a given prompt.

Usage:
    1. Create an instance of the `MarkovChain` class.
    2. Train the model using the `train` method with a text corpus.
    3. Generate new text using the `generate` method with a prompt and desired length.

Example:
    >>> mc = MarkovChain()
    >>> mc.train("This is a simple example. This example is simple.")
    >>> print(mc.generate("This", length=5))
    This is simple example is

Dependencies:
    - random: Used for randomly selecting the next word during text generation.
    - string.punctuation: Used for removing punctuation during tokenization.
    - collections.defaultdict: Used for storing the Markov Chain graph as a dictionary of lists.

Notes:
    - The `_tokenize` method removes punctuation and numbers, converts newlines to spaces, and splits text into words.
    - The `train` method builds a graph where each word points to a list of possible next words.
    - The `generate` method uses the graph to create a sequence of words, starting from the last word in the prompt.

Limitations:
    - The model assumes that the input text is well-formed and does not handle edge cases like empty input gracefully.
    - The generated text may not always be coherent, as it relies purely on word transitions without considering grammar or context.
"""

# The `random` module is used to randomly select the next word during text generation.
# Specifically, the `random.choice` method is used to pick a word from the list of possible next words.
import random

# The `punctuation` constant from the `string` module provides a list of all punctuation characters.
# It is used in the `_tokenize` method to remove punctuation from the input text during tokenization.
from string import punctuation

# The `defaultdict` class from the `collections` module is used to create the Markov Chain graph.
# It initializes the graph as a dictionary where each key (a word) maps to a list of possible next words.
# This simplifies the process of appending new words to the graph without needing to check for key existence.import random
from collections import defaultdict

class MarkovChain:
    def __init__(self):
        """
        Initializes the MarkovChain instance.

        This constructor sets up the Markov Chain graph as a `defaultdict` of lists. 
        Each key in the graph represents a word, and the corresponding value is a list 
        of words that can follow it based on the training data.

        Attributes:
            graph (defaultdict): A dictionary where each key is a word and the value 
                                 is a list of possible next words.
        """
        self.graph = defaultdict(list)

    def _tokenize(self, text):
        """
        Tokenizes the input text by removing punctuation, numbers, and splitting it into words.

        This method processes the input text to prepare it for training or generation. It removes
        all punctuation and numeric characters, replaces newlines with spaces, and splits the text
        into individual words.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of words (tokens) extracted from the input text.

        Notes:
            - Punctuation and numbers are removed using `str.maketrans` and `str.translate`.
            - Newlines are replaced with spaces to ensure consistent tokenization.
            - The resulting text is split into words using the `split` method.
        """
        return (
            text.translate(str.maketrans("", "", punctuation + "1234567890"))
            .replace("\n", " ")
            .split(" ")
        )
    
    def train(self, text):
        """
        Trains the Markov Chain model by building a graph of word transitions from the input text.

        This method processes the input text, tokenizes it into words, and constructs a graph where 
        each word points to a list of possible next words based on the sequence in the input text.

        Args:
            text (str): The input text used to train the Markov Chain model.

        Notes:
            - The `_tokenize` method is used to preprocess the input text by removing punctuation, 
            numbers, and splitting it into words.
            - The graph is constructed as a `defaultdict` where each key is a word, and the value 
            is a list of words that can follow it in the input text.
            - The method iterates through the tokenized words and appends the next word in the sequence 
            to the list of possible transitions for the current word.

        Example:
            >>> mc = MarkovChain()
            >>> mc.train("This is a simple example. This example is simple.")
            >>> print(mc.graph)
            defaultdict(<class 'list'>, {'This': ['is'], 'is': ['a', 'simple.'], 'a': ['simple'], 
                                        'simple': ['example.'], 'example.': ['This'], 'example': ['is']})
        """
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            if (len(tokens) - 1) == i:
                break
            self.graph[token].append(tokens[i + 1])
               

    def generate(self, prompt, length=10):
        """
        Generates a sequence of text based on the trained Markov Chain and a given prompt.

        This method uses the Markov Chain graph to generate a sequence of words. It starts with the 
        last word in the given prompt and iteratively selects the next word based on the possible 
        transitions in the graph. The process continues until the desired sequence length is reached.

        Args:
            prompt (str): The initial text to start the generation. The last word of the prompt is 
                        used as the starting point for the generation.
            length (int): The number of words to generate in the sequence (default is 10).

        Returns:
            str: A string containing the generated sequence of text.

        Notes:
            - The `_tokenize` method is used to extract the last word from the prompt.
            - The `random.choice` method is used to randomly select the next word from the list of 
            possible transitions for the current word.
            - If no transitions are available for the current word, the generation process skips to 
            the next iteration without adding a new word.
            - The generated text is appended to the initial prompt to form the final output.

        Example:
            >>> mc = MarkovChain()
            >>> mc.train("This is a simple example. This example is simple.")
            >>> print(mc.generate("This", length=5))
            This is a simple example
        """
        # Get the last token from the prompt
        current = self._tokenize(prompt)[-1]
        # Initialize the output with the prompt
        output = prompt
        for i in range(length):
            # Look up the options in the graph dictionary
            options = self.graph.get(current, [])
            if not options:
                continue
            # Use random.choice to pick the next word
            current = random.choice(options)
            # Add the selected word to the output
            output += f" {current}"
        
        return output
    

# Usage example

text = """
Andrey Markov was born on 14 June 1856 in Russia. He attended the St. Petersburg Grammar School, where some teachers saw him as a rebellious student. In his academics he performed poorly in most subjects other than mathematics. Later in life he attended Saint Petersburg Imperial University (now Saint Petersburg State University). Among his teachers were Yulian Sokhotski (differential calculus, higher algebra), Konstantin Posse (analytic geometry), Yegor Zolotarev (integral calculus), Pafnuty Chebyshev (number theory and probability theory), Aleksandr Korkin (ordinary and partial differential equations), Mikhail Okatov (mechanism theory), Osip Somov (mechanics), and Nikolai Budajev (descriptive and higher geometry). He completed his studies at the university and was later asked if he would like to stay and have a career as a Mathematician. He later taught at high schools and continued his own mathematical studies. In this time he found a practical use for his mathematical skills. He figured out that he could use chains to model the alliteration of vowels and consonants in Russian literature. He also contributed to many other mathematical aspects in his time. He died at age 66 on 20 July 1922.
Torvalds was born in Helsinki, Finland, the son of journalists Anna and Nils Torvalds,[7] the grandson of statistician Leo Törnqvist and of poet Ole Torvalds, and the great-grandson of journalist and soldier Toivo Karanko. His parents were campus radicals at the University of Helsinki in the 1960s. His family belongs to the Swedish-speaking minority in Finland. He was named after Linus Pauling, the Nobel Prize-winning American chemist, although in the book Rebel Code: Linux and the Open Source Revolution, he is quoted as saying, "I think I was named equally for Linus the Peanuts cartoon character", noting that this made him "half Nobel Prize-winning chemist and half blanket-carrying cartoon character".[8]

Torvalds attended the University of Helsinki from 1988 to 1996,[9] graduating with a master's degree in computer science from the NODES research group.[10] His academic career was interrupted after his first year of study when he joined the Finnish Navy Nyland Brigade in the summer of 1989, selecting the 11-month officer training program to fulfill the mandatory military service of Finland. He gained the rank of second lieutenant, with the role of an artillery observer.[11] He bought computer science professor Andrew Tanenbaum's book Operating Systems: Design and Implementation, in which Tanenbaum describes MINIX, an educational stripped-down version of Unix. In 1990, Torvalds resumed his university studies, and was exposed to Unix for the first time in the form of a DEC MicroVAX running ULTRIX.[12] His MSc thesis was titled Linux: A Portable Operating System.[13]

His interest in computers began with a VIC-20[14] at the age of 11 in 1981. He started programming for it in BASIC, then later by directly accessing the 6502 CPU in machine code (he did not utilize assembly language).[15] He then purchased a Sinclair QL, which he modified extensively, especially its operating system. "Because it was so hard to get software for it in Finland", he wrote his own assembler and editor "(in addition to Pac-Man graphics libraries)"[16] for the QL, and a few games.[17][18] He wrote a Pac-Man clone, Cool Man. On 5 January 1991[19] he purchased an Intel 80386-based clone of IBM PC[20] before receiving his MINIX copy, which in turn enabled him to begin work on Linux.

Linux
Main article: History of Linux
The first Linux prototypes were publicly released in late 1991.[8][21] Version 1.0 was released on 14 March 1994.[22]

Torvalds first encountered the GNU Project in 1991 when another Swedish-speaking computer science student, Lars Wirzenius, took him to the University of Technology to listen to free software guru Richard Stallman's speech.[citation needed] Torvalds used Stallman's GNU General Public License version 2 (GPLv2) for his Linux kernel.

After a visit to Transmeta in late 1996,[23] Torvalds accepted a position at the company in California, where he worked from February 1997 to June 2003. He then moved to the Open Source Development Labs, which has since merged with the Free Standards Group to become the Linux Foundation, under whose auspices he continues to work. In June 2004, Torvalds and his family moved to Dunthorpe, Oregon[24] to be closer to the OSDL's headquarters in Beaverton.

From 1997 to 1999, he was involved in 86open, helping select the standard binary format for Linux and Unix. In 1999, he was named by the MIT Technology Review TR100 as one of the world's top 100 innovators under age 35.[25]

In 1999, Red Hat and VA Linux, both leading developers of Linux-based software, presented Torvalds with stock options in gratitude for his creation.[26] That year both companies went public and Torvalds's share value briefly shot up to about US$20 million.[27][28]

His personal mascot is a penguin nicknamed Tux,[29] which has been widely adopted by the Linux community as the Linux kernel's mascot.[30]

Although Torvalds believes "open source is the only right way to do software", he also has said that he uses the "best tool for the job", even if that includes proprietary software.[31] He was criticized for his use and alleged advocacy of the proprietary BitKeeper software for version control in the Linux kernel. He subsequently wrote a free-software replacement for it called Git.

In 2008, Torvalds stated that he used the Fedora Linux distribution because it had fairly good support for the PowerPC processor architecture, which he favored at the time.[32] He confirmed this in a 2012 interview.[33] He has also posted updates about his choice of desktop environment, often in response to perceived feature regressions.

The Linux Foundation currently sponsors Torvalds so he can work full-time on improving Linux.[34]

Torvalds is known for vocally disagreeing with other developers on the Linux kernel mailing list.[35] Calling himself a "really unpleasant person", he explained, "I'd like to be a nice person and curse less and encourage people to grow rather than telling them they are idiots. I'm sorry—I tried, it's just not in me."[36][37] His attitude, which he considers necessary for making his points clear, has drawn criticism from Intel programmer Sage Sharp and systemd developer Lennart Poettering, among others.[38][failed verification][39]

On Sunday, 16 September 2018, the Linux kernel Code of Conflict was suddenly replaced by a new Code of Conduct based on the Contributor Covenant. Shortly thereafter, in the release notes for Linux 4.19-rc4, Torvalds apologized for his behavior, calling his personal attacks of the past "unprofessional and uncalled for" and announced a period of "time off" to "get some assistance on how to understand people's emotions and respond appropriately". It soon transpired that these events followed The New Yorker approaching Torvalds with a series of questions critical of his conduct.[40][41][42] Following the release of Linux 4.19 on 22 October 2018, Torvalds returned to maintaining the kernel.[43]
"""

chain = MarkovChain()
chain.train(text)
sample_prompt = "He was"
print(chain.generate(sample_prompt))

result = chain.generate(sample_prompt)