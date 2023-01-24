package seakers.eossrl.actions.partitioningandassignment;

import seakers.vassarexecheur.search.operators.partitioning.RepairInterferencePartitioning;
import java.util.HashMap;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator;

public class RepairInterference extends RepairInterferencePartitioning {
    
    private int numChanges; // Move size in the design space
    private HashMap<String, String[]> interferenceMap; // mapping of instruments to their interfering instrument sets
    private PartitioningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairInterference(int numChanges, HashMap<String, String[]> interferenceMap, PartitioningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(numChanges, resourcePool, evaluator, params, problem, interferenceMap);
        this.numChanges = numChanges;
        this.interferenceMap = interferenceMap;
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