package seakers.eossrl.actions.partitioningandassignment;

import seakers.vassarexecheur.search.operators.partitioning.RepairDutyCyclePartitioning;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator;

public class RepairDutyCycle extends RepairDutyCyclePartitioning {

    private final double threshold; // Minimum duty cycle to satisfy heuristic
    private final int numChanges; // number of moves in design space
    private PartitioningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairDutyCycle(double threshold, int numChanges, PartitioningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(threshold, numChanges, params, problem, resourcePool, evaluator);
        this.threshold = threshold;
        this.numChanges = numChanges;
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