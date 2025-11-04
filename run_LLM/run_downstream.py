from run_LLM.modules import Main
from run_LLM.downstream_model_class.data_classes import DownstreamTrainArgs

if __name__ == "__main__":
    run_config = DownstreamTrainArgs()
    main_runner = Main(run_config, category="Baby_Products")
    main_runner.main()