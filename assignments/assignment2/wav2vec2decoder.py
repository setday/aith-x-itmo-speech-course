from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """

        transcript = []  # Prepare empty list for transcript tokens
        prev_char = None # Previous character ID

        for logit in logits:                       # We don't need to use softmax here, as we are using argmax directly
            max_index = torch.argmax(logit).item() # Get the index of the most probable token

            if max_index != self.blank_token_id and max_index != prev_char: # Skip blank tokens and duplicates
                transcript.append(self.vocab[max_index])                    # Append the token to the transcript
                prev_char = max_index                                       # Update the previous character ID

        return ''.join(transcript).replace(self.word_delimiter, ' ').strip() # Join the tokens and remove word delimiters

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        
        probs = torch.softmax(logits, dim=-1)                   # Apply softmax to get probabilities
        beams = [(torch.ones(1), '', self.blank_token_id)] # Initialize beams with a single empty hypothesis

        for layer_probs in probs: # Iterate over each time step
            new_beams = {}        # Dictionary to store new beams

            for next_id, proba in enumerate(layer_probs):                                         # Iterate over each character ID and its probability
                for beam_prob, hypothesis, last_char_id in beams:                                 # Iterate over existing beams
                    if last_char_id != next_id and next_id != self.blank_token_id:                # Only non-blank and non-duplicate characters can be added to the hypothesis
                        new_char = self.vocab[next_id] if next_id != self.word_delimiter else ' ' # Handle blank token 
                        hypothesis = hypothesis + new_char                                        # Append new character ID to the hypothesis

                    key = (hypothesis, next_id)                        # Create a key for the new beam
                    acc_beam_prob = new_beams.get(key, torch.zeros(1)) # Get the existing beam probability

                    new_beams[key] = acc_beam_prob + proba * beam_prob # Update the probability of the new beam

            new_beams = [(prob, hyp, last_char) for (hyp, last_char), prob in new_beams.items()] # Convert dictionary to list of tuples
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:self.beam_width]        # Keep only the top beam_width beams

        beams = [(hyp, torch.log(prob + 1e-10).item()) for prob, hyp, _ in beams] # Convert beams to log probabilities
        transcript = max(beams, key=lambda x: x[0])[0]                            # Get the best hypothesis based on log probability

        best_hypothesis = ''.join(transcript).replace(self.word_delimiter, ' ').strip() # Join the tokens and remove word delimiters

        if return_beams:
            return beams
        else:
            return best_hypothesis

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
        
        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        
        probs = torch.softmax(logits, dim=-1)                   # Apply softmax to get probabilities
        beams = [(torch.ones(1), '', self.blank_token_id)] # Initialize beams with a single empty hypothesis

        for layer_probs in probs: # Iterate over each time step
            new_beams = {}        # Dictionary to store new beams

            for next_id, proba in enumerate(layer_probs):                                         # Iterate over each character ID and its probability
                for beam_prob, hypothesis, last_char_id in beams:                                 # Iterate over existing beams
                    if last_char_id != next_id and next_id != self.blank_token_id:                # Only non-blank and non-duplicate characters can be added to the hypothesis
                        new_char = self.vocab[next_id] if next_id != self.word_delimiter else ' ' # Handle blank token 
                        hypothesis = hypothesis + new_char                                        # Append new character ID to the hypothesis

                    key = (hypothesis, next_id)                        # Create a key for the new beam
                    acc_beam_prob = new_beams.get(key, torch.zeros(1)) # Get the existing beam probability
                    
                    _, last_word = hypothesis.rsplit(' ', 1) if ' ' in hypothesis else (None, '') # Get the last word from the hypothesis
                    lm_prob = torch.tensor(self.lm_model.score(last_word, bos=True, eos=False)) # Get the LM score for the hypothesis

                    new_beams[key] = acc_beam_prob + proba * beam_prob + self.alpha * lm_prob # Update the probability of the new beam

            new_beams = [(prob, hyp, last_char) for (hyp, last_char), prob in new_beams.items()] # Convert dictionary to list of tuples
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:self.beam_width]        # Keep only the top beam_width beams

        best_beam = max(beams, key=lambda x: x[0]) # Get the best beam based on log probability

        return ''.join(best_beam[1]).replace(self.word_delimiter, ' ').strip() # Join the tokens and remove word delimiters

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
        
        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        rescored_beams = []
        for beam in beams:
            hypothesis, log_prob = beam
            lm_prob = self.lm_model.score(hypothesis)
            rescored_prob = log_prob + self.alpha * lm_prob
            rescored_beams.append((hypothesis, rescored_prob))
            
        best_hypothesis = max(rescored_beams, key=lambda x: x[1])[0] # Get the best hypothesis based on rescored probability

        return ''.join(best_hypothesis).replace(self.word_delimiter, ' ').strip() # Join the tokens and remove word delimiters

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
