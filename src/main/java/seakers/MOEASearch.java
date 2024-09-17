package seakers;

import org.moeaframework.algorithm.AbstractEvolutionaryAlgorithm;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import seakers.aos.aos.AOS;
import seakers.aos.history.AOSHistoryIO;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.Callable;

public class MOEASearch implements Callable<Algorithm> {
    private Algorithm algorithm;
    private String saveDirectory;
    private int maximumNFE;
    private int runNumber;
    private final String[] variableNames;
    private final String[] objectiveNames;
    private final String[] constraintNames;
    private final String[] heuristicNames;
    private final boolean isPartitioning;
    private final boolean isEOSS;

    public MOEASearch(Algorithm algorithm, String saveDirectory, int maximumNFE, int runNumber, String[] variableNames, String[] objectiveNames, String[] constraintNames, String[] heuristicNames, boolean isPartitioning, boolean isEOSS) {
        this.algorithm = algorithm;
        this.saveDirectory = saveDirectory;
        this.maximumNFE = maximumNFE;
        this.runNumber = runNumber;
        this.variableNames = variableNames;
        this.objectiveNames = objectiveNames;
        this.constraintNames = constraintNames;
        this.heuristicNames = heuristicNames;
        this.isPartitioning = isPartitioning;
        this.isEOSS = isEOSS;
    }

    @Override
    public Algorithm call() throws Exception {
        System.out.println("Starting MOEA Run" + runNumber);

        Result result = new Result(saveDirectory);

        HashSet<Solution> exploredSolutions = new HashSet<>();

        ArrayList<Solution> allSolutions = new ArrayList<>();

        long startTime = System.currentTimeMillis();
        int currentNumberOfFunctionEvaluations = 0;
        algorithm.step();

        Population initialPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
        for (Solution solution : initialPopulation) {
            allSolutions.add(solution);
            exploredSolutions.add(solution);
            solution.setAttribute("NFE", currentNumberOfFunctionEvaluations);
        }
        currentNumberOfFunctionEvaluations = initialPopulation.size();

        while (!algorithm.isTerminated() && (algorithm.getNumberOfEvaluations() < maximumNFE)) {
            algorithm.step();
            Population currentPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
            for (int i = 1; i < 3; i++) { // Only valid for Epsilon MOEA
                Solution currentSolution = currentPopulation.get(currentPopulation.size() - i);
                if (!exploredSolutions.contains(currentSolution)){
                    currentNumberOfFunctionEvaluations++;
                    exploredSolutions.add(currentSolution);
                }
                currentSolution.setAttribute("NFE", currentNumberOfFunctionEvaluations);
                allSolutions.add(currentSolution);
            }

            if ((algorithm.getNumberOfEvaluations() % 100 == 0) && ((algorithm.getProblem() instanceof AssigningProblem) || (algorithm.getProblem() instanceof PartitioningProblem))) {
                System.out.println("NFE = " + algorithm.getNumberOfEvaluations());
                if (algorithm.getProblem() instanceof AssigningProblem) {
                    ((AssigningProblem) algorithm.getProblem()).getEvaluationManager().getResourcePool().poolClean();
                } else if (algorithm.getProblem() instanceof PartitioningProblem) {
                    ((PartitioningProblem) algorithm.getProblem()).getEvaluationManager().getResourcePool().poolClean();
                }

            }
        }

        algorithm.terminate();
        long endTime = System.currentTimeMillis();

        System.out.println("Total Execution Time: " + ((endTime - startTime)/1000.0) + " s");

        Population finalPopulation = ((AbstractEvolutionaryAlgorithm) algorithm).getPopulation();
        NondominatedPopulation finalArchive = ((AbstractEvolutionaryAlgorithm) algorithm).getArchive();

        String algorithmName = "EpsilonMOEA_";
        if (algorithm instanceof AOS) {
            algorithmName = "AOSMOEA_";
        }

        if (isEOSS) {
            if (isPartitioning) {
                ((PartitioningProblem) algorithm.getProblem()).getEvaluationManager().clear();
            } else {
                ((AssigningProblem) algorithm.getProblem()).getEvaluationManager().clear();
            }
        }

        // save final population
        result.saveInternalPopulationOrArchive( algorithmName + runNumber + "_finalpop.csv", finalPopulation, variableNames, objectiveNames, constraintNames, heuristicNames, isPartitioning, isEOSS);

        // save final archive
        result.saveInternalPopulationOrArchive(algorithmName + runNumber + "_finalarchive.csv", finalArchive, variableNames, objectiveNames, constraintNames, heuristicNames, isPartitioning, isEOSS);

        // save all solutions
        result.saveAllInternalSolutions(algorithmName + runNumber + "_allSolutions.csv", allSolutions, variableNames, objectiveNames, constraintNames, heuristicNames, isPartitioning, isEOSS);

        // save AOS credit, quality and selection histories (if applicable)
        if (algorithm instanceof AOS) {
            AOS algAOS = (AOS) algorithm;
            AOSHistoryIO.saveQualityHistory(algAOS.getQualityHistory(), new File(saveDirectory + File.separator + algorithmName + "_" + runNumber + "_qual.csv"), ",");
            AOSHistoryIO.saveCreditHistory(algAOS.getCreditHistory(), new File(saveDirectory + File.separator + algorithmName + "_" + runNumber + "_credit.csv"), ",");
            AOSHistoryIO.saveSelectionHistory(algAOS.getSelectionHistory(), new File(saveDirectory + File.separator + algorithmName + "_" + runNumber + "_hist.csv"), ",");
        }
        return algorithm;
    }
}
