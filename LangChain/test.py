"""
Brief explanation:
This tiny script checks whether the LangChain package is installed and prints the version
currently available in the environment.

Author's note:
This utility is useful as a quick environment sanity check before running larger LangChain examples.
"""

import langchain

# Print the installed LangChain version to confirm the environment is set up correctly.
print(langchain.__version__)
