from run_LLM.modules import Main
from run_LLM.downstream_model_class.data_classes import DownstreamTrainArgs

if __name__ == "__main__":
    run_config = DownstreamTrainArgs()

    pretrain_categories = [
        "Video_Games",
        "Arts_Crafts_and_Sewing",
        "Movies_and_TV",
        # "Home_and_Kitchen",
        "Electronics",
        "Tools_and_Home_Improvement",
    ]

    outofdomain_categories = ["Baby_Products", "Sports_and_Outdoors"]

    for cat in pretrain_categories + outofdomain_categories:
        main_runner = Main(run_config, category=cat)
        main_runner.main()