package seakers;

import com.mathworks.engine.MatlabEngine;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.algorithm.EpsilonMOEA;
import org.moeaframework.core.*;
import org.moeaframework.core.comparator.AggregateConstraintComparator;
import org.moeaframework.core.comparator.ChainedComparator;
import org.moeaframework.core.comparator.DominanceComparator;
import org.moeaframework.core.comparator.ParetoObjectiveComparator;
import org.moeaframework.core.operator.*;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.operator.real.UM;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.problem.CDTLZ.C1_DTLZ1;
import org.moeaframework.problem.CEC2009.UF1;
import seakers.aos.aos.AOSMOEA;
import seakers.aos.creditassignment.offspringparent.OffspringParentDomination;
import seakers.aos.creditassignment.setimprovement.SetImprovementDominance;
import seakers.aos.operator.AOSVariation;
import seakers.aos.operator.AOSVariationOP;
import seakers.aos.operator.AOSVariationSI;
import seakers.aos.operatorselectors.AdaptivePursuit;
import seakers.aos.operatorselectors.OperatorSelector;
import seakers.aos.operatorselectors.ProbabilityMatching;
import seakers.trussaos.operators.constantradii.AddDiagonalMember;
import seakers.trussaos.operators.constantradii.AddMember;
import seakers.trussaos.operators.constantradii.ImproveOrientation2;
import seakers.trussaos.operators.constantradii.RemoveIntersection2;
import seakers.trussaos.problems.ConstantRadiusArteryProblem;
import seakers.trussaos.problems.ConstantRadiusTrussProblem2;
import seakers.vassarexecheur.search.intialization.SynchronizedMersenneTwister;
import seakers.vassarexecheur.search.intialization.partitioning.RandomPartitioningReadInitialization;
import seakers.vassarexecheur.search.operators.assigning.*;
import seakers.vassarexecheur.search.operators.partitioning.*;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.evaluation.AbstractArchitectureEvaluator;
import seakers.vassarheur.evaluation.ArchitectureEvaluationManager;
import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;
import seakers.vassarheur.problems.Assigning.ClimateCentricAssigningParams;
import seakers.vassarheur.problems.PartitioningAndAssigning.ClimateCentricPartitioningParams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.*;

public class MOEARun {

    public static ExecutorService pool;
    public static CompletionService<Algorithm> cs;
    /**
     * Matlab Engine for function evaluation
     */
    private static MatlabEngine engine;

    public static void main (String[] args) throws InterruptedException, ExecutionException {

        // Save location
        //String saveDir = System.getProperty("user.dir") + File.separator + "results"; // File not found error!
        String saveDir = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\results";

        // Problem parameters
        int populationSize = 840;
        int maximumEvaluations = 42000;

        int numCPU = 4;
        int numRuns = 10; // comment if lines 73 and 74 are uncommented

        // Use next two lines for Partitioning problem where specific architectures need to be initialized for specific runs
        //int[] runNumbers = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}; // change numCPU accordingly
        //int numRuns = runNumbers.length; // comment lines 408 and 532 and uncomment lines 409 and 533

        pool = Executors.newFixedThreadPool(numCPU);
        cs = new ExecutorCompletionService<>(pool);

        double crossoverProbability = 1.0;

        // Test problem
        int problemChoice = 2; // Problem choice = 0 -> C1_DTLZ1, 1 -> UF1, 2 -> Either Metamaterial problem, 3 -> Either EOSS problem
        boolean arteryProblem = false; // Only useful if problemChoice is 2
        boolean assigningProblem = true; // Only useful if problemChoice is 3

        // Defined for all problems
        boolean[][] heuristicsConstrained = new boolean[0][];
        ArrayList<Boolean> aosConstrained = new ArrayList<>();
        String[] variableNames = new String[0];
        String[] objectiveNames = new String[0];
        String[] constraintNames = new String[0];
        String[] allHeuristicNames = new String[0];
        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        double[] epsilonBox = new double[0];
        Initialization initialization = null;
        Variation crossover = null;
        Variation mutation = null;

        double mutationProbability = 0.0;

        // Defined only for satellite problems
        String[] instrumentList = new String[0];
        String[] orbitsList = new String[0];
        String resourcesPath = "C:\\SEAK Lab\\SEAK Lab Github\\VASSAR\\VASSAR_resources-heur"; // for lab system

        double dcThreshold = 0.5;
        double massThreshold = 3000.0; // [kg]
        double packEffThreshold = 0.7;
        double instrCountThreshold = 15; // only for assigning problem
        boolean considerFeasibility = true;

        PRNG.setRandom(new SynchronizedMersenneTwister());

        AbstractProblem problem = null;
        Variation[] heuristicOperators = new Variation[0];
        AOSVariation aosStrategy = null;
        switch (problemChoice) {

            case 0:
                problem = new C1_DTLZ1(2);
                variableNames = new String[problem.getNumberOfVariables()];
                objectiveNames = new String[problem.getNumberOfObjectives()];
                constraintNames = new String[problem.getNumberOfConstraints()];

                for (int i = 0; i < variableNames.length; i++) {
                    variableNames[i] = "Variable" + i;
                }

                for (int i = 0; i < objectiveNames.length; i++) {
                    objectiveNames[i] = "Objective" + i;
                }

                for (int i = 0; i < constraintNames.length; i++) {
                    constraintNames[i] = "Constraint" + i;
                }

                epsilonBox = new double[problem.getNumberOfObjectives()];
                Arrays.fill(epsilonBox, 0.0001);

                mutationProbability = 1.0/problem.getNumberOfVariables();
                crossover = new UniformCrossover(crossoverProbability);
                mutation = new UM(mutationProbability);

                break;

            case 1:
                problem = new UF1();
                variableNames = new String[problem.getNumberOfVariables()];
                objectiveNames = new String[problem.getNumberOfObjectives()];
                constraintNames = new String[problem.getNumberOfConstraints()];

                for (int i = 0; i < variableNames.length; i++) {
                    variableNames[i] = "Variable" + i;
                }

                for (int i = 0; i < objectiveNames.length; i++) {
                    objectiveNames[i] = "Objective" + i;
                }

                for (int i = 0; i < constraintNames.length; i++) {
                    constraintNames[i] = "Constraint" + i;
                }

                epsilonBox = new double[problem.getNumberOfObjectives()];
                Arrays.fill(epsilonBox, 0.0001);

                mutationProbability = 1.0/problem.getNumberOfVariables();
                crossover = new UniformCrossover(crossoverProbability);
                mutation = new UM(mutationProbability);

                break;

            case 2:
                engine = MatlabEngine.startMatlab();

                String matlabScriptsLocation = "C:\\SEAK Lab\\SEAK Lab Github\\Heuristics in RL\\matlab";
                engine.feval("addpath", matlabScriptsLocation); // Add location of MATLAB scripts used to compute objectives, constraints and heuristics to MATLAB's search path

                /**
                 * modelChoice = 0 --> Fibre Stiffness Model
                 *             = 1 --> Truss Stiffness Model
                 *             = 2 --> Beam Model
                 */
                int modelChoice = 1; // Fibre stiffness model cannot be used for the artery problem

                double targetStiffnessRatio = 1;
                if (arteryProblem) {
                    targetStiffnessRatio = 0.421;
                }

                // For AOS MOEA Run
                boolean maintainFeasibility = false;

                // Heuristic Enforcement Methods
                /**
                 * partialCollapsibilityConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
                 * nodalPropertiesConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
                 * orientationConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
                 * intersectionConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint, AHS]
                 *
                 * heuristicsConstrained = [partialCollapsibilityConstrained, nodalPropertiesConstrained, orientationConstrained, intersectionConstrained]
                 */
                boolean[] partialCollapsibilityConstrained = {false, false, false, false, false, false, false};
                boolean[] nodalPropertiesConstrained = {false, false, false, false, false, false, false};
                boolean[] orientationConstrained = {false, false, false, false, false, false, false};
                boolean[] intersectionConstrained = {false, false, false, false, false, false, false};

                heuristicsConstrained = new boolean[4][7];
                for (int i = 0; i < 7; i++) {
                    heuristicsConstrained[0][i] = partialCollapsibilityConstrained[i];
                    heuristicsConstrained[1][i] = nodalPropertiesConstrained[i];
                    heuristicsConstrained[2][i] = orientationConstrained[i];
                    heuristicsConstrained[3][i] = intersectionConstrained[i];
                }

                for (int i = 0; i < 4; i++) {
                    aosConstrained.add(heuristicsConstrained[i][1]);
                }

                numberOfHeuristicConstraints = 0;
                numberOfHeuristicObjectives = 0;
                for (int i = 0; i < 4; i++) {
                    if (heuristicsConstrained[i][5]) {
                        numberOfHeuristicConstraints++;
                    }
                    if (heuristicsConstrained[i][4]) {
                        numberOfHeuristicObjectives++;
                    }
                }

                // New dimensions for printable solutions
                double printableRadius = 250e-6; // in m
                double printableSideLength = 10e-3; // in m
                double printableModulus = 1.8162e6; // in Pa
                double sideNodeNumber = 5.0D;
                int nucFactor = 3; // Not used if PBC model is used

                int totalNumberOfMembers;
                if (sideNodeNumber >= 5) {
                    int sidenumSquared = (int) (sideNodeNumber*sideNodeNumber);
                    totalNumberOfMembers =  sidenumSquared * (sidenumSquared - 1)/2;
                }
                else {
                    totalNumberOfMembers = (int) (CombinatoricsUtils.factorial((int) (sideNodeNumber*sideNodeNumber))/(CombinatoricsUtils.factorial((int) ((sideNodeNumber*sideNodeNumber) - 2)) * CombinatoricsUtils.factorial(2)));
                }
                int numberOfRepeatableMembers = (int) (2 * (CombinatoricsUtils.factorial((int) sideNodeNumber)/(CombinatoricsUtils.factorial((int) (sideNodeNumber - 2)) * CombinatoricsUtils.factorial(2))));
                int numVariables = totalNumberOfMembers - numberOfRepeatableMembers;

                double[][] globalNodePositions;
                if (arteryProblem) {
                    problem = new ConstantRadiusArteryProblem(saveDir, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, engine, heuristicsConstrained);
                    globalNodePositions = ((ConstantRadiusArteryProblem) problem).getNodalConnectivityArray();
                } else {
                    problem = new ConstantRadiusTrussProblem2(saveDir, modelChoice, numVariables, numberOfHeuristicObjectives, numberOfHeuristicConstraints, printableRadius, printableSideLength, printableModulus, sideNodeNumber, nucFactor, targetStiffnessRatio, engine, heuristicsConstrained);
                    globalNodePositions = ((ConstantRadiusTrussProblem2) problem).getNodalConnectivityArray();
                }
                variableNames = new String[problem.getNumberOfVariables()];
                objectiveNames = new String[problem.getNumberOfObjectives()];
                constraintNames = new String[problem.getNumberOfConstraints()];

                for (int i = 0; i < variableNames.length; i++) {
                    variableNames[i] = "Variable" + i;
                }

                for (int i = 0; i < objectiveNames.length; i++) {
                    objectiveNames[i] = "TrueObjective" + (i+1);
                }

                if (arteryProblem) {
                    constraintNames = new String[]{"FeasibilityViolation", "ConnectivityViolation"};
                } else {
                    constraintNames = new String[]{"FeasibilityViolation", "ConnectivityViolation", "StiffnessRatioViolation"};
                }
                allHeuristicNames = new String[]{"PartialCollapsibilityViolation","NodalPropertiesViolation","OrientationViolation","IntersectionViolation"};

                epsilonBox = new double[problem.getNumberOfObjectives()];
                Arrays.fill(epsilonBox, 0.0001);

                mutationProbability = 1.0/problem.getNumberOfVariables();
                crossover = new OnePointCrossover(crossoverProbability);
                mutation = new BitFlip(mutationProbability);

                Variation addMember = new CompoundVariation(new AddMember(maintainFeasibility, arteryProblem, engine, globalNodePositions, sideNodeNumber, printableSideLength, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));
                Variation removeIntersection = new CompoundVariation(new RemoveIntersection2(arteryProblem, engine, globalNodePositions, sideNodeNumber, printableSideLength, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));
                Variation addDiagonalMember = new CompoundVariation(new AddDiagonalMember(maintainFeasibility, arteryProblem, engine, globalNodePositions, sideNodeNumber, printableSideLength, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));
                Variation improveOrientation = new CompoundVariation(new ImproveOrientation2(arteryProblem, globalNodePositions, targetStiffnessRatio, (int) sideNodeNumber, numberOfHeuristicObjectives, numberOfHeuristicConstraints), new BitFlip(mutationProbability));

                heuristicOperators = new Variation[]{addDiagonalMember, addMember, improveOrientation, removeIntersection};

                break;

            case 3:

                // Heuristic Enforcement Methods
                /**
                 * dutyCycleConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 * instrumentOrbitRelationsConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 * interferenceConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 * packingEfficiencyConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 * spacecraftMassConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 * synergyConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 * instrumentCountConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
                 *
                 * if partitioning problem:
                 * heuristicsConstrained = [dutyCycleConstrained, instrumentOrbitRelationsConstrained, interferenceConstrained, packingEfficiencyConstrained, spacecraftMassConstrained, synergyConstrained]
                 * else:
                 * heuristicsConstrained = [dutyCycleConstrained, instrumentOrbitRelationsConstrained, interferenceConstrained, packingEfficiencyConstrained, spacecraftMassConstrained, synergyConstrained, instrumentCountConstrained]
                 */
                boolean[] dutyCycleConstrained = {false, true, false, false, false, false};
                boolean[] instrumentOrbitRelationsConstrained = {false, true, false, false, false, false};
                boolean[] interferenceConstrained = {false, true, false, false, false, false};
                boolean[] packingEfficiencyConstrained = {false, true, false, false, false, false};
                boolean[] spacecraftMassConstrained = {false, true, false, false, false, false};
                boolean[] synergyConstrained = {false, true, false, false, false, false};
                boolean[] instrumentCountConstrained = {false, true, false, false, false, false}; // only for assigning problem

                if (assigningProblem) {
                    heuristicsConstrained = new boolean[7][6];
                } else {
                    heuristicsConstrained = new boolean[6][6];
                }

                for (int i = 0; i < heuristicsConstrained[0].length; i++) {
                    heuristicsConstrained[0][i] = dutyCycleConstrained[i];
                    heuristicsConstrained[1][i] = instrumentOrbitRelationsConstrained[i];
                    heuristicsConstrained[2][i] = interferenceConstrained[i];
                    heuristicsConstrained[3][i] = packingEfficiencyConstrained[i];
                    heuristicsConstrained[4][i] = spacecraftMassConstrained[i];
                    heuristicsConstrained[5][i] = synergyConstrained[i];
                    if (assigningProblem) {
                        heuristicsConstrained[6][i] = instrumentCountConstrained[i];
                    }
                }

                for (boolean[] booleans : heuristicsConstrained) {
                    aosConstrained.add(booleans[1]);
                }

                numberOfHeuristicConstraints = 0;
                numberOfHeuristicObjectives = 0;
                for (int i = 0; i < heuristicsConstrained.length; i++) {
                    if (heuristicsConstrained[i][5]) {
                        numberOfHeuristicConstraints++;
                    }
                    if (heuristicsConstrained[i][4]) {
                        numberOfHeuristicObjectives++;
                    }
                }

                constraintNames = new String[]{};
                if (assigningProblem) {
                    allHeuristicNames = new String[]{"DCViolation","InstrOrbViolation","InterInstrViolation","PackEffViolation","SpMassViolation","SynergyViolation","InstrCountViolation"};
                } else {
                    allHeuristicNames = new String[]{"DCViolation","InstrOrbViolation","InterInstrViolation","PackEffViolation","SpMassViolation","SynergyViolation"};
                }
                break;

            default:
                System.out.println("No problem chosen");
        }

        DominanceComparator comparator;
        if ((problemChoice == 1) || (problemChoice == 3)) { // UF1 and EOSS problems are unconstrained
            comparator = new ParetoObjectiveComparator();
        } else {
            comparator = new ChainedComparator(new AggregateConstraintComparator(), new ParetoObjectiveComparator());
        }

        Variation variation = null;
        EpsilonBoxDominanceArchive archive;
        Selection selection = new TournamentSelection(2, comparator);

        for (int i = 0; i < numRuns; i++) {
            Population population = new Population();

            BaseParams params;
            AbstractArchitectureEvaluator evaluator;
            HashMap<String, String[]> instrumentSynergyMap;
            HashMap<String, String[]> interferingInstrumentsMap;

            Algorithm moea = null;

            if (problemChoice == 3) { // EOSS Problems

                if (assigningProblem) {
                    params = new ClimateCentricAssigningParams(resourcesPath, "FUZZY-ATTRIBUTES","test", "normal");
                    instrumentSynergyMap = getInstrumentSynergyNameMap(params);
                    interferingInstrumentsMap = getInstrumentInterferenceNameMap(params);
                    evaluator = new ArchitectureEvaluator(considerFeasibility, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold);
                } else {
                    params = new ClimateCentricPartitioningParams(resourcesPath, "FUZZY-ATTRIBUTES", "test", "normal");
                    instrumentSynergyMap = getInstrumentSynergyNameMap(params);
                    interferingInstrumentsMap = getInstrumentInterferenceNameMap(params);
                    evaluator = new seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator(considerFeasibility, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold);
                }

                ArchitectureEvaluationManager evaluationManager = new ArchitectureEvaluationManager(params, evaluator);
                evaluationManager.init(1);

                if (assigningProblem) {
                    problem = new AssigningProblem(new int[]{1}, params.getProblemName(), evaluationManager, (ArchitectureEvaluator) evaluator, params, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold, instrCountThreshold, numberOfHeuristicObjectives, numberOfHeuristicConstraints, heuristicsConstrained);
                } else {
                    problem = new PartitioningProblem(params.getProblemName(), evaluationManager, params, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold, numberOfHeuristicObjectives, numberOfHeuristicConstraints, heuristicsConstrained);
                }

                if (!assigningProblem) {
                    initialization = new RandomPartitioningReadInitialization(saveDir, i, populationSize, (PartitioningProblem) problem, instrumentList, orbitsList);
                    //initialization = new RandomPartitioningReadInitialization(saveDir, runNumbers[i], populationSize, (PartitioningProblem) problem, instrumentList, orbitsList);
                } else {
                    initialization = new RandomInitialization(problem, populationSize);
                }

                epsilonBox = new double[problem.getNumberOfObjectives()];
                Arrays.fill(epsilonBox, 0.0001);
                archive = new EpsilonBoxDominanceArchive(epsilonBox);

                if (assigningProblem) {
                    crossover = new OnePointCrossover(crossoverProbability);
                    mutation = new BitFlip(mutationProbability);
                } else {
                    crossover = new PartitioningCrossover(crossoverProbability, params);
                    mutation = new PartitioningMutation(mutationProbability, params);
                }

                if (aosConstrained.contains(true)) { // If AOS MOEA Run
                    // Initialize heuristic operators
                    Variation repairDutyCycle;
                    Variation repairInstrumentOrbitRelations;
                    Variation repairInterference;
                    Variation repairPackingEfficiency;
                    Variation repairMass;
                    Variation repairSynergy;
                    Variation repairInstrumentCount = null;

                    if (assigningProblem) { // duty Cycle, interference, mass -> remove, instrument orbit -> move, pack Eff, synergy -> Add
                        repairDutyCycle = new CompoundVariation(new RepairDutyCycleAssigning(dcThreshold, 1, params, false, (AssigningProblem) problem, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
                        repairInstrumentOrbitRelations = new CompoundVariation(new RepairInstrumentOrbitAssigning(1, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) problem, true), new BitFlip(mutationProbability));
                        repairInterference = new CompoundVariation(new RepairInterferenceAssigning(1, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) problem, interferingInstrumentsMap, false), new BitFlip(mutationProbability));
                        repairPackingEfficiency = new CompoundVariation(new RepairPackingEfficiencyAdditionAssigning(packEffThreshold, 1, 1, params, (AssigningProblem) problem, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
                        repairMass = new CompoundVariation(new RepairMassAssigning(massThreshold, 1, params, false, (AssigningProblem) problem, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
                        repairSynergy = new CompoundVariation(new RepairSynergyAdditionAssigning(1, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) problem, instrumentSynergyMap), new BitFlip(mutationProbability));
                        repairInstrumentCount = new CompoundVariation(new RepairInstrumentCountAssigning(1, 1, instrCountThreshold, (AssigningProblem) problem, evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params), new BitFlip(mutationProbability));
                    } else {
                        repairDutyCycle = new CompoundVariation(new RepairDutyCyclePartitioning(dcThreshold, 1, params, (PartitioningProblem) problem, evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator), new PartitioningMutation(mutationProbability, params));
                        repairInstrumentOrbitRelations = new CompoundVariation(new RepairInstrumentOrbitPartitioning(1, evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator, params, (PartitioningProblem) problem), new PartitioningMutation(mutationProbability, params));
                        repairInterference = new CompoundVariation(new RepairInterferencePartitioning(1, evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator, params, (PartitioningProblem) problem, interferingInstrumentsMap), new PartitioningMutation(mutationProbability, params));
                        repairPackingEfficiency = new CompoundVariation(new RepairPackingEfficiencyPartitioning(packEffThreshold, 1, params, (PartitioningProblem) problem, evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator), new PartitioningMutation(mutationProbability, params));
                        repairMass = new CompoundVariation(new RepairMassPartitioning(massThreshold, 1, params, (PartitioningProblem) problem, evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator), new PartitioningMutation(mutationProbability, params));
                        repairSynergy = new CompoundVariation(new RepairSynergyPartitioning(1, evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator, params, (PartitioningProblem) problem, instrumentSynergyMap), new PartitioningMutation(mutationProbability, params));
                    }

                    heuristicOperators = new Variation[]{repairDutyCycle, repairInstrumentOrbitRelations, repairInterference, repairPackingEfficiency, repairMass, repairSynergy, repairInstrumentCount};

                    ArrayList<Variation> operators = new ArrayList<>();
                    for (int k = 0; k < heuristicsConstrained.length; k++) {
                        if (heuristicsConstrained[k][1]) {
                            operators.add(heuristicOperators[k]);
                        }
                    }

                    operators.add(new CompoundVariation(crossover, mutation));

                    // Create operator selector
                    OperatorSelector operatorSelector = new AdaptivePursuit(operators, 0.8, 0.8, 0.1);

                    // Create credit assignment
                    SetImprovementDominance creditAssignment = new SetImprovementDominance(archive, 1, 0);

                    // Create AOS
                    aosStrategy = new AOSVariationSI(operatorSelector, creditAssignment, populationSize);

                    // Creating AOS MOEA object
                    EpsilonMOEA emoea = new EpsilonMOEA(problem, population, archive, selection, aosStrategy, initialization, comparator);
                    moea = new AOSMOEA(emoea, aosStrategy, true);
                } else { // Simple Epsilon MOEA Run
                    variation = new CompoundVariation(crossover, mutation);
                    moea = new EpsilonMOEA(problem, population, archive, selection, variation, initialization, comparator);
                }

                instrumentList = params.getInstrumentList();
                orbitsList = params.getOrbitList();
                mutationProbability = 1.0/problem.getNumberOfVariables();

                if (assigningProblem) {
                    variableNames = getVariableNames(problem.getNumberOfVariables() - 1); // Not counting the initial integer variable set in the AssigningArchitecture (which does not contribute to the architecture evaluation but present for legacy reasons)
                } else {
                    variableNames = getVariableNames(problem.getNumberOfVariables());
                }
                objectiveNames = getObjectiveNames(problem.getNumberOfObjectives());

            } else if (problemChoice == 2) { // Metamaterial Problems

                initialization = new RandomInitialization(problem, populationSize);

                archive = new EpsilonBoxDominanceArchive(epsilonBox);

                if (aosConstrained.contains(true)) { // AOS MOEA Run
                    ArrayList<Variation> operators = new ArrayList<>();
                    for (int k = 0; k < heuristicsConstrained.length; k++) {
                        if (heuristicsConstrained[k][1]) {
                            operators.add(heuristicOperators[k]);
                        }
                    }

                    operators.add(new CompoundVariation(new OnePointCrossover(crossoverProbability), new BitFlip(mutationProbability)));

                    // Create operator selector
                    OperatorSelector operatorSelector = new ProbabilityMatching(operators, 0.6, 0.03);

                    // Create credit assignment
                    OffspringParentDomination creditAssignment = new OffspringParentDomination(1.0, 0.5, 0.0, comparator);

                    // Create AOS
                    aosStrategy = new AOSVariationOP(operatorSelector, creditAssignment, populationSize);

                    // Creating AOS MOEA object
                    EpsilonMOEA emoea = new EpsilonMOEA(problem, population, archive, selection, aosStrategy, initialization, comparator);
                    moea = new AOSMOEA(emoea, aosStrategy, true);

                } else { // Simple Epsilon MOEA Run
                    variation = new CompoundVariation(crossover, mutation);
                    moea = new EpsilonMOEA(problem, population, archive, selection, variation, initialization, comparator);
                }

            } else { // Other problems do not have repair operators, so just simple Epsilon MOEA
                initialization = new RandomInitialization(problem, populationSize);
                archive = new EpsilonBoxDominanceArchive(epsilonBox);
                moea = new EpsilonMOEA(problem, population, archive, selection, variation, initialization, comparator);
            }

            cs.submit(new MOEASearch(moea, saveDir, maximumEvaluations, i, variableNames, objectiveNames, constraintNames, allHeuristicNames, (problemChoice == 3) && (!assigningProblem), (problemChoice == 3)));
            //cs.submit(new MOEASearch(moea, saveDir, maximumEvaluations, runNumbers[i], variableNames, objectiveNames, constraintNames, allHeuristicNames, (problemChoice == 3) && (!assigningProblem), (problemChoice == 3)));
        }

        for (int i = 0; i < numRuns; i++) {
            try {
                cs.take().get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        pool.shutdown();
    }

    /**
     * Creates instrument synergy map used to compute the instrument synergy violation heuristic (only formulated for the
     * Climate Centric problem for now) (added by roshansuresh)
     * @param params
     * @return Instrument synergy hashmap
     */
    protected static HashMap<String, String[]> getInstrumentSynergyNameMap(BaseParams params) {
        HashMap<String, String[]> synergyNameMap = new HashMap<>();
        if (params.getProblemName().equalsIgnoreCase("ClimateCentric")) {
            synergyNameMap.put("ACE_ORCA", new String[]{"DESD_LID", "GACM_VIS", "ACE_POL", "HYSP_TIR", "ACE_LID"});
            synergyNameMap.put("DESD_LID", new String[]{"ACE_ORCA", "ACE_LID", "ACE_POL"});
            synergyNameMap.put("GACM_VIS", new String[]{"ACE_ORCA", "ACE_LID"});
            synergyNameMap.put("HYSP_TIR", new String[]{"ACE_ORCA", "POSTEPS_IRS"});
            synergyNameMap.put("ACE_POL", new String[]{"ACE_ORCA", "DESD_LID"});
            synergyNameMap.put("ACE_LID", new String[]{"ACE_ORCA", "CNES_KaRIN", "DESD_LID", "GACM_VIS"});
            synergyNameMap.put("POSTEPS_IRS", new String[]{"HYSP_TIR"});
            synergyNameMap.put("CNES_KaRIN", new String[]{"ACE_LID"});
        }
        else {
            System.out.println("Synergy Map for current problem not formulated");
        }
        return synergyNameMap;
    }

    /**
     * Creates instrument interference map used to compute the instrument interference violation heuristic (only formulated for the
     * Climate Centric problem for now)
     * @param params
     * @return Instrument interference hashmap
     */
    protected static HashMap<String, String[]> getInstrumentInterferenceNameMap(BaseParams params) {
        HashMap<String, String[]> interferenceNameMap = new HashMap<>();
        if (params.getProblemName().equalsIgnoreCase("ClimateCentric")) {
            interferenceNameMap.put("ACE_LID", new String[]{"ACE_CPR", "DESD_SAR", "CLAR_ERB", "GACM_SWIR"});
            interferenceNameMap.put("ACE_CPR", new String[]{"ACE_LID", "DESD_SAR", "CNES_KaRIN", "CLAR_ERB", "ACE_POL", "ACE_ORCA", "GACM_SWIR"});
            interferenceNameMap.put("DESD_SAR", new String[]{"ACE_LID", "ACE_CPR"});
            interferenceNameMap.put("CLAR_ERB", new String[]{"ACE_LID", "ACE_CPR"});
            interferenceNameMap.put("CNES_KaRIN", new String[]{"ACE_CPR"});
            interferenceNameMap.put("ACE_POL", new String[]{"ACE_CPR"});
            interferenceNameMap.put("ACE_ORCA", new String[]{"ACE_CPR"});
            interferenceNameMap.put("GACM_SWIR", new String[]{"ACE_LID", "ACE_CPR"});
        }
        else {
            System.out.println("Interference Map fpr current problem not formulated");
        }
        return interferenceNameMap;
    }

    static String[] getVariableNames(int numberOfVariables) {
        String[] variableNames = new String[numberOfVariables];

        for (int i = 0; i < variableNames.length; i++) {
            variableNames[i] = "Variable" + i;
        }
        return variableNames;
    }

    static String[] getObjectiveNames(int numberOfObjectives) {
        String[] objectiveNames = new String[numberOfObjectives]; // Attributes names for the unpenalized objectives as recorded in solutions

        for (int i = 0; i < objectiveNames.length; i++) {
            objectiveNames[i] = "TrueObjective" + (i + 1);
        }

        return objectiveNames;
    }
}
