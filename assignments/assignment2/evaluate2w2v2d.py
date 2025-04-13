import torchaudio
from evaluate import load

from tqdm import tqdm

import tabulate

from wav2vec2decoder import Wav2Vec2Decoder

def test(decoder, audio_path, true_transcription, results):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"
    
    wer_metric = load("wer")

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        transcript = decoder.decode(audio_input, method=d_strategy)
        l_dist = Levenshtein.distance(true_transcription, transcript.strip())
        wer = wer_metric.compute(predictions=[transcript], references=[true_transcription])
        
        prev_l_dist, prev_wer = results[d_strategy]
        results[d_strategy] = (l_dist + prev_l_dist, wer + prev_wer)


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

    lm_model_path: str = "lm/3-gram.pruned.1e-7.arpa.gz" # lm/4-gram.arpa.gz
    decoder = Wav2Vec2Decoder(lm_model_path=lm_model_path, beam_width=5)

    # results = []
    # for beam_width in [4, 6, 7, 8, 9]:
    #     decoder.beam_width = beam_width
        
    #     result = {"beam_width": beam_width, "greedy": (0, 0), "beam": (0, 0), "beam_lm": (0, 0), "beam_lm_rescore": (0, 0)}
    #     _ = [test(decoder, audio_path, target, result) for audio_path, target in tqdm(test_samples)]
        
    #     for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
    #         l_dist, wer = result[d_strategy]
    #         l_dist, wer = l_dist / len(test_samples), wer / len(test_samples)
    #         result[d_strategy] = f"{l_dist} / {wer:.2%}"

    #     results.append(result)
    # decoder.beam_width = 3
        
    # print(tabulate.tabulate(results, tablefmt="grid", headers="keys"))

    # results = []
    # for alpha in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    #     decoder.alpha = alpha
        
    #     result = {"alpha": alpha, "greedy": (0, 0), "beam": (0, 0), "beam_lm": (0, 0), "beam_lm_rescore": (0, 0)}
    #     _ = [test(decoder, audio_path, target, result) for audio_path, target in tqdm(test_samples)]
        
    #     for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
    #         l_dist, wer = result[d_strategy]
    #         l_dist, wer = l_dist / len(test_samples), wer / len(test_samples)
    #         result[d_strategy] = f"{l_dist} / {wer:.2%}"

    #     results.append(result)
    # decoder.alpha = 1.0
        
    # print(tabulate.tabulate(results, tablefmt="grid", headers="keys"))

    results = []
    for beta in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        decoder.beta = beta
        
        result = {"beta": beta, "greedy": (0, 0), "beam": (0, 0), "beam_lm": (0, 0), "beam_lm_rescore": (0, 0)}
        _ = [test(decoder, audio_path, target, result) for audio_path, target in tqdm(test_samples)]
        
        for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
            l_dist, wer = result[d_strategy]
            l_dist, wer = l_dist / len(test_samples), wer / len(test_samples)
            result[d_strategy] = f"{l_dist} / {wer:.2%}"

        results.append(result)
    decoder.beta = 1.0
        
    print(tabulate.tabulate(results, tablefmt="grid", headers="keys"))
