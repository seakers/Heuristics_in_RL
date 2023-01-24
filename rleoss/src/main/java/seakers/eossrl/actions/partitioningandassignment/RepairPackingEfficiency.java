package seakers.eossrl.actions.partitioningandassignment;

import seakers.vassarexecheur.search.operators.partitioning.RepairPackingEfficiencyPartitioning;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator;

public class RepairPackingEfficiency extends RepairPackingEfficiencyPartitioning {
    
    private final double threshold; // Minimum packing efficiency to satisfy heuristic
    private int numInstruments; // number of instruments to move
    private PartitioningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairPackingEfficiency(double threshold, int numInstruments, PartitioningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(threshold, numInstruments, params, problem, resourcePool, evaluator);
        this.threshold = threshold;
        this.numInstruments = numInstruments;
        this.problem = problem;
        this.resourcePool = resourcePool;
        this.params = params;
        this.evaluator = evaluator;
    }

    // All parent methods can be used, below are modified versions of the methods to override them if needed

    /*
     * Methods: 
     *  1. Solution[] evolve(Solution[] sols)
     *  2. int getArity()
     */

}