from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        
        chars = list(text) # make a list of tokens
        chars_ = sorted(set(chars)) # sort the tokens in alphabetical order
        stoi = {o:i for i, o in enumerate(chars_)} # populate a dictionary of integer-token pairs
        itos = {i:o for o, i in stoi.items()} # populate a dictionary of token-integer pairs

        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        
        return [stoi[t] for t in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        
        return ''.join([itos[i] for i in ids])