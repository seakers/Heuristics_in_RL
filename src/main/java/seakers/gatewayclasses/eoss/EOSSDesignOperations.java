package seakers.gatewayclasses.eoss;

import org.moeaframework.core.PRNG;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variable;
import org.moeaframework.core.Variation;
import org.moeaframework.core.operator.CompoundVariation;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.EncodingUtils;
import org.moeaframework.problem.AbstractProblem;
import seakers.architecture.util.IntegerVariable;
import seakers.gatewayclasses.DesignOperations;
import seakers.vassarexecheur.search.intialization.SynchronizedMersenneTwister;
import seakers.vassarexecheur.search.operators.assigning.*;
import seakers.vassarexecheur.search.operators.partitioning.*;
import seakers.vassarexecheur.search.problems.assigning.AssigningArchitecture;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningArchitecture;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.evaluation.AbstractArchitectureEvaluator;
import seakers.vassarheur.evaluation.ArchitectureEvaluationManager;
import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;
import seakers.vassarheur.problems.Assigning.ClimateCentricAssigningParams;
import seakers.vassarheur.problems.PartitioningAndAssigning.ClimateCentricPartitioningParams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

public class EOSSDesignOperations extends DesignOperations {

    // Parameters for problem class
    private String resourcesPath;
    private boolean considerFeasibility;
    private String[] instrumentNames;
    private String[] orbitNames;
    private boolean assigningProblem;
    private AbstractProblem problem;
    private ArchitectureEvaluationManager evaluationManager;
    private int numCPU;

    private int evaluationCount;

    // Parameters for RL action and reward computation
    private ArrayList<String> objectiveNames;
    private ArrayList<String> heuristicNames;
    private ArrayList<Integer> actions;
    // For assigning problem, list only contains one integer representing which design variable (corresponding to the instrument in the satellite) to add or remove
    // For partitioning problem, list contains two integers, one representing which instrument to move and the second representing which orbit to move it to
    private ArrayList<Boolean> heuristicsDeployed;
    private ArrayList<Variation> heuristicOperators;
    private double dutyCycleThreshold;
    private double massThreshold;
    private double packingEfficiencyThreshold;
    private double instrumentCountThreshold;
    private ArrayList<Double> objectives;
    private ArrayList<Double> heuristics;
    private ArrayList<Integer> currentDesignDecisions;
    private Solution currentDesign;
    private ArrayList<ArrayList<String>> currentDesignPayloads;
    private ArrayList<String> currentDesignOrbits;
    private ArrayList<Integer> newDesignDecisions;

    /**
     * Constructor for EOSS operations class instance
     * sets the problem instance, modifies a design based on the action and evaluates a design
     */

    public EOSSDesignOperations() {

        this.numCPU = 1; // For Architecture Evaluation Manager
        this.evaluationCount = 0;
        PRNG.setRandom(new SynchronizedMersenneTwister());
    }

    // Run parameter setting methods before initializing problem instance parameter
    public void setHeuristicThresholds(double dcThreshold, double massThreshold, double packEffThreshold, double instrCountThreshold) {
        this.dutyCycleThreshold = dcThreshold;
        this.massThreshold = massThreshold;
        this.packingEfficiencyThreshold = packEffThreshold;
        this.instrumentCountThreshold = instrCountThreshold;
        //System.out.println("duty cycle threshold = " + dcThreshold);
        //System.out.println("mass threshold = " + massThreshold);
        //System.out.println("packing efficiency threshold = " + packEffThreshold);
        //System.out.println("instrument count threshold = " + instrCountThreshold);
    }

    // Run parameter setting methods before initializing problem instance parameter
    public void setConsiderFeasibility(boolean considerFeasibility) {
        this.considerFeasibility = considerFeasibility;
        //System.out.println("consider feasibility = " + considerFeasibility);
    }

    // Run parameter setting methods before initializing problem instance parameter
    public void setAssigningProblem(boolean assigningProblem) {
        this.assigningProblem = assigningProblem;
        //System.out.println("assigning problem = " + assigningProblem);
    }


    // Run parameter setting methods before initializing problem instance parameter
    public void setResourcesPath(String resourcesPath) {
        this.resourcesPath = resourcesPath;
        //System.out.println("resources path = " + resourcesPath);
    }

    // Setting method for heuristics deployed
    public void setHeuristicsDeployed(ArrayList<Boolean> heuristicsDeployed) {
        this.heuristicsDeployed = heuristicsDeployed;
        //System.out.println("Heuristics deployed = " + heuristicsDeployed);
    }

    // Setting methods for data saving
    public void setObjectiveNames(ArrayList<String> objectiveNames) {
        this.objectiveNames = objectiveNames;
        //System.out.println("Objective Names = " + objectiveNames);
    }

    public void setHeuristicNames(ArrayList<String> heuristicNames) {
        this.heuristicNames = heuristicNames;
        //System.out.println("heuristic names = " + heuristicNames);
    }

    public void setProblem() {
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
        boolean[] dutyCycleConstrained = {false, false, false, false, false, false};
        boolean[] instrumentOrbitRelationsConstrained = {false, true, false, false, false, false};
        boolean[] interferenceConstrained = {false, true, false, false, false, false};
        boolean[] packingEfficiencyConstrained = {false, false, false, false, false, false};
        boolean[] spacecraftMassConstrained = {false, true, false, false, false, false};
        boolean[] synergyConstrained = {false, false, false, false, false, false};
        boolean[] instrumentCountConstrained = {false, true, false, false, false, false}; // only for assigning problem

        boolean[][] heuristicsConstrained;
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

        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        for (int i = 0; i < heuristicsConstrained.length; i++) {
            if (heuristicsConstrained[i][5]) {
                numberOfHeuristicConstraints++;
            }
            if (heuristicsConstrained[i][4]) {
                numberOfHeuristicObjectives++;
            }
        }

        double mutationProbability;
        BaseParams params;
        AbstractArchitectureEvaluator evaluator;
        HashMap<String, String[]> instrumentSynergyMap;
        HashMap<String, String[]> interferingInstrumentsMap;
        if (this.assigningProblem) {
            mutationProbability = 1. / 60.;
            params = new ClimateCentricAssigningParams(this.resourcesPath, "FUZZY-ATTRIBUTES","test", "normal");

            instrumentSynergyMap = getInstrumentSynergyNameMap(params);
            interferingInstrumentsMap = getInstrumentInterferenceNameMap(params);

            evaluator = new ArchitectureEvaluator(this.considerFeasibility, interferingInstrumentsMap, instrumentSynergyMap, this.dutyCycleThreshold, this.massThreshold, this.packingEfficiencyThreshold);
        } else {
            mutationProbability = 1. / 24.; // Based on the 12 instruments for the ClimateCentric Problem
            params = new ClimateCentricPartitioningParams(this.resourcesPath, "FUZZY-ATTRIBUTES", "test", "normal");

            instrumentSynergyMap = getInstrumentSynergyNameMap(params);
            interferingInstrumentsMap = getInstrumentInterferenceNameMap(params);

            evaluator = new seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator(this.considerFeasibility, interferingInstrumentsMap, instrumentSynergyMap, this.dutyCycleThreshold, this.massThreshold, this.packingEfficiencyThreshold);
        }

        this.instrumentNames = params.getInstrumentList();
        this.orbitNames = params.getOrbitList();

        this.evaluationManager = new ArchitectureEvaluationManager(params, evaluator);
        this.evaluationManager.init(this.numCPU);

        // Problem class
        AbstractProblem problem;
        if (this.assigningProblem) {
            this.problem = new AssigningProblem(new int[]{1}, params.getProblemName(), this.evaluationManager, (ArchitectureEvaluator) evaluator, params, interferingInstrumentsMap, instrumentSynergyMap, this.dutyCycleThreshold, this.massThreshold, this.packingEfficiencyThreshold, this.instrumentCountThreshold, numberOfHeuristicObjectives, numberOfHeuristicConstraints, heuristicsConstrained);
        } else {
            this.problem = new PartitioningProblem(params.getProblemName(), this.evaluationManager, params, interferingInstrumentsMap, instrumentSynergyMap, this.dutyCycleThreshold, this.massThreshold, this.packingEfficiencyThreshold, numberOfHeuristicObjectives, numberOfHeuristicConstraints, heuristicsConstrained);
        }

        // Initialize heuristic operators
        Variation repairDutyCycle;
        Variation repairInstrumentOrbitRelations;
        Variation repairInterference;
        Variation repairPackingEfficiency;
        Variation repairMass;
        Variation repairSynergy;
        Variation repairInstrumentCount = null;

        if (assigningProblem) { // duty Cycle, interference, mass -> remove, instrument orbit -> move, pack Eff, synergy -> Add
            repairDutyCycle = new CompoundVariation(new RepairDutyCycleAssigning(this.dutyCycleThreshold, 1, params, false, (AssigningProblem) this.problem, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
            repairInstrumentOrbitRelations = new CompoundVariation(new RepairInstrumentOrbitAssigning(1, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) this.problem, true), new BitFlip(mutationProbability));
            repairInterference = new CompoundVariation(new RepairInterferenceAssigning(1, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) this.problem, interferingInstrumentsMap, false), new BitFlip(mutationProbability));
            //repairPackingEfficiency = new CompoundVariation(new RepairPackingEfficiencyAssigning(this.packingEfficiencyThreshold, 1, params, false, (AssigningProblem) this.problem, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
            repairPackingEfficiency = new CompoundVariation(new RepairPackingEfficiencyAdditionAssigning(this.packingEfficiencyThreshold, 1, 1, params, (AssigningProblem) this.problem, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
            repairMass = new CompoundVariation(new RepairMassAssigning(this.massThreshold, 1, params, false, (AssigningProblem) this.problem, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator), new BitFlip(mutationProbability));
            //repairSynergy = new CompoundVariation(new RepairSynergyAssigning(1, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) this.problem, instrumentSynergyMap, false), new BitFlip(mutationProbability));
            repairSynergy = new CompoundVariation(new RepairSynergyAdditionAssigning(1, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params, (AssigningProblem) this.problem, instrumentSynergyMap), new BitFlip(mutationProbability));
            repairInstrumentCount = new CompoundVariation(new RepairInstrumentCountAssigning(1, 1, this.instrumentCountThreshold, (AssigningProblem) this.problem, this.evaluationManager.getResourcePool(), (ArchitectureEvaluator) evaluator, params), new BitFlip(mutationProbability));
        } else {
            repairDutyCycle = new CompoundVariation(new RepairDutyCyclePartitioning(this.dutyCycleThreshold, 1, params, (PartitioningProblem) this.problem, this.evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator), new PartitioningMutation(mutationProbability, params));
            repairInstrumentOrbitRelations = new CompoundVariation(new RepairInstrumentOrbitPartitioning(1, this.evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator, params, (PartitioningProblem) this.problem), new PartitioningMutation(mutationProbability, params));
            repairInterference = new CompoundVariation(new RepairInterferencePartitioning(1, this.evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator, params, (PartitioningProblem) this.problem, interferingInstrumentsMap), new PartitioningMutation(mutationProbability, params));
            repairPackingEfficiency = new CompoundVariation(new RepairPackingEfficiencyPartitioning(this.packingEfficiencyThreshold, 1, params, (PartitioningProblem) this.problem, this.evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator), new PartitioningMutation(mutationProbability, params));
            repairMass = new CompoundVariation(new RepairMassPartitioning(this.massThreshold, 1, params, (PartitioningProblem) this.problem, this.evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator), new PartitioningMutation(mutationProbability, params));
            repairSynergy = new CompoundVariation(new RepairSynergyPartitioning(1, this.evaluationManager.getResourcePool(), (seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator) evaluator, params, (PartitioningProblem) this.problem, instrumentSynergyMap), new PartitioningMutation(mutationProbability, params));
        }

        Variation[] allHeuristicOperators = {repairDutyCycle, repairInstrumentOrbitRelations, repairInterference, repairPackingEfficiency, repairMass, repairSynergy, repairInstrumentCount};

        ArrayList<Variation> deployedHeuristicOperators = new ArrayList<>();
        for (int j = 0; j < allHeuristicOperators.length; j++) {
            if (this.heuristicsDeployed.get(j)) {
                deployedHeuristicOperators.add(allHeuristicOperators[j]);
            }
        }

        this.heuristicOperators = deployedHeuristicOperators;
        //System.out.println("Problem set");

    }

    // Run problem setting methods before running this method
    public void setCurrentDesign(ArrayList<Integer> design) {
        this.currentDesignDecisions = design;
        Solution currentSolution = this.problem.newSolution();

        for (int i = 1; i < currentSolution.getNumberOfVariables(); ++i) {
            if (this.assigningProblem) {
                BinaryVariable var = new BinaryVariable(1);
                if (design.get(i-1) == 1) {
                    EncodingUtils.setBoolean(var, true);
                } else {
                    EncodingUtils.setBoolean(var, false);
                }
                currentSolution.setVariable(i, var);
            } else {
                IntegerVariable var;
                if (i < this.instrumentNames.length) {
                    var = new IntegerVariable(design.get(i-1), 0, this.instrumentNames.length);
                } else {
                    var = new IntegerVariable(design.get(i-1), -1, this.instrumentNames.length);
                }
                currentSolution.setVariable(i, var);
            }
        }

        this.currentDesign = currentSolution;

        //System.out.println("");
        //System.out.println("Current design set");
    }

    public void setAction(ArrayList<Integer> actions) {
        this.actions = actions;
        //System.out.println("Actions = " + actions);
    }

    // Method to clear Rete object (run method periodically to avoid running out of heap memory)
    public void clearRete() {
        this.evaluationManager.getResourcePool().poolClean();
        System.out.println("Rete clean initiated");
    }

    // Set current design before evaluating
    public ArrayList<Double> evaluate() {
        this.problem.evaluate(this.currentDesign);

        // Results are stored in the order -> {objectives, constraints, heuristics}
        ArrayList<Double> designMetrics = new ArrayList<>();
        ArrayList<Double> currentObjectives = new ArrayList<>();
        for (int i = 0; i < this.currentDesign.getNumberOfObjectives(); i++) {
            currentObjectives.add(this.currentDesign.getObjective(i));
        }

        ArrayList<Double> currentHeuristics = new ArrayList<>();
        for (String name : this.heuristicNames) {
            currentHeuristics.add((double) this.currentDesign.getAttribute(name));
        }

        this.objectives = currentObjectives;
        this.heuristics = currentHeuristics;

        designMetrics.addAll(currentObjectives);
        designMetrics.addAll(currentHeuristics);

        if (this.assigningProblem) {
            this.currentDesignPayloads = ((AssigningArchitecture) this.currentDesign).getSatellitePayloads();
            this.currentDesignOrbits = ((AssigningArchitecture) this.currentDesign).getSatelliteOrbits();
        } else {
            this.currentDesignPayloads = ((PartitioningArchitecture) this.currentDesign).getSatellitePayloads();
            this.currentDesignOrbits = ((PartitioningArchitecture) this.currentDesign).getSatelliteOrbits();
        }
        this.evaluationCount++;

        System.out.println("Design evaluation complete. " + this.evaluationCount);
        //System.out.println("Objectives = " + currentObjectives);
        //System.out.println("Heuristics = " + currentHeuristics);
        //System.out.println("");

        return designMetrics;
    }

    public void operate() {
        ArrayList<Integer> newDesignDecisions = new ArrayList<>();

        ////// NEW ACTION SPACE: n_actions = n_states + n_heurs
        if (this.assigningProblem) {
            newDesignDecisions = new ArrayList<>(this.currentDesignDecisions);
            if (this.actions.get(0) < (this.instrumentNames.length*this.orbitNames.length)) { // Simple bit-flipping of design decisions (only one design decision for assigning problem)
                newDesignDecisions.set(this.actions.get(0), 1 - (this.currentDesignDecisions.get(this.actions.get(0))));
            } else { // heuristic actions
                Variation selectedHeuristicOperator = this.heuristicOperators.get(this.actions.get(0) - (this.instrumentNames.length*this.orbitNames.length));
                //System.out.println("Action = " + action + ", Heuristic Operator  = " + ((CompoundVariation) selectedHeuristicOperator).getName());
                Solution newSolution = selectedHeuristicOperator.evolve(new Solution[]{this.currentDesign})[0];
                for (int i = 0; i < newSolution.getNumberOfVariables(); i++) {
                    newDesignDecisions.set(i, EncodingUtils.getInt(newSolution.getVariable(i)));
                }
            }
        } else { // Partitioning problem
            // TBD
            System.out.println("To be coded later");
        }

        this.newDesignDecisions = newDesignDecisions;
        //System.out.println("Operation complete. New design = " + newDesignDecisions);
        //System.out.println("");
    }

    // Run operate() before getting new design
    public int[] getNewDesign() {
        return this.newDesignDecisions.stream().mapToInt(i->i).toArray();
    }

    // Run setProblem() before obtaining instrument and orbit names
    public ArrayList<String> getInstrumentNames() {
        return new ArrayList<>(Arrays.asList(this.instrumentNames));
    }

    public ArrayList<String> getOrbitNames() {
        return new ArrayList<>(Arrays.asList(this.orbitNames));
    }

    // Get objectives and heuristics (run evaluate() before these)
    public ArrayList<Double> getObjectives() {
        return this.objectives;
    }

    public ArrayList<Double> getTrueObjectives() {
        ArrayList<Double> trueObjectives = new ArrayList<>();
        for (String objectiveName : this.objectiveNames) {
            trueObjectives.add((Double) this.currentDesign.getAttribute(objectiveName));
        }
        return trueObjectives;
    }

    public ArrayList<Double> getHeuristics() {
        return this.heuristics;
    }

    // Reset method for new design evaluation and manipulation
    public void resetDesignGoals() {
        this.objectives = new ArrayList<>();
        this.heuristics = new ArrayList<>();
        this.currentDesign = null;
        this.currentDesignDecisions = new ArrayList<>();
        this.newDesignDecisions = new ArrayList<>();
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






}
