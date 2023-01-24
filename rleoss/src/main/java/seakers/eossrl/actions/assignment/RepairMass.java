package seakers.eossrl.actions.assignment;

import seakers.vassarexecheur.search.operators.assigning.RepairMassAssigning;
import seakers.vassarheur.ResourcePool;
import seakers.vassarexecheur.search.problems.assigning.AssigningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.problems.Assigning.ArchitectureEvaluator;

public class RepairMass extends RepairMassAssigning {
    
    private final double threshold; // Maximum spacecraft mass to satisfy heuristic
    private final int numChanges; // number of moves in design space
    private boolean moveInstruments; // true -> change = move instruments, false -> change = remove instruments
    private AssigningProblem problem; // problem class for architecture evaluation
    private final ResourcePool resourcePool; // used for architecture evaluation
    private final BaseParams params; // used for architecture evaluation
    private final ArchitectureEvaluator evaluator; // used for architecture evaluation

    public RepairMass(double threshold, int numChanges, boolean moveInstruments, AssigningProblem problem, ResourcePool resourcePool, BaseParams params, ArchitectureEvaluator evaluator) {
        super(threshold, numChanges, params, moveInstruments, problem, resourcePool, evaluator);
        this.threshold = threshold;
        this.numChanges = numChanges;
        this.moveInstruments = moveInstruments;
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