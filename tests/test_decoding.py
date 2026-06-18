from keta.decoding.mbr import chrf_score_simple
from keta.decoding.metrics import DialectScorer


def test_chrf_score_simple():
    same = chrf_score_simple("شلونك يا ولد عساك بخير", "شلونك يا ولد عساك بخير")
    assert same > 0.95

    diff = chrf_score_simple("شلونك يا ولد عساك بخير", "الآن سوف نذهب للمكتبة")
    assert diff < 0.3


def test_dialect_scorer():
    scorer = DialectScorer()
    gulf = scorer.score("تكفى شلونك الحين؟ وش سويت في الشحنة وايد تأخرت")
    msa = scorer.score("الآن سوف نذهب إلى المنزل ونريد معرفة ماذا تفعل")
    assert gulf > msa


if __name__ == "__main__":
    test_chrf_score_simple()
    test_dialect_scorer()
    print("test_decoding.py passed")
