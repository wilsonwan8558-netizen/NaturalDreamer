import gymnasium as gym
import torch
import argparse
import os
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    
    env             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName), (64, 64))))
    envEvaluation   = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName, render_mode="rgb_array"), (64, 64))))
    
    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}")

    print("Initializing Dreamer...")
    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config.dreamer)
    print(f"Dreamer initialized on device: {device}")
    if config.resume:
        print(f"Loading checkpoint: {checkpointToLoad}")
        dreamer.loadCheckpoint(checkpointToLoad)

    print(f"Collecting initial episodes: {config.episodesBeforeStart}")
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)
    print(f"Initial collection done. envSteps={dreamer.totalEnvSteps}, episodes={dreamer.totalEpisodes}")

    iterationsNum = config.gradientSteps // config.replayRatio
    print(f"Starting training: gradientSteps={config.gradientSteps}, replayRatio={config.replayRatio}, checkpointInterval={config.checkpointInterval}")
    for _ in range(iterationsNum):
        for _ in range(config.replayRatio):
            sampledData                         = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData)
            behaviorMetrics                     = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1
            if dreamer.totalGradientSteps % 10 == 0:
                print(f"gradientSteps={dreamer.totalGradientSteps}, envSteps={dreamer.totalEnvSteps}")

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                print(f"Saving checkpoint: {checkpointFilenameBase}_{suffix}.pth")
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    main(parser.parse_args().config)
