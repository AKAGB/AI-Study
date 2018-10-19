(define (domain blocks)

(:requirements :strips :typing :equality :conditional-effects :universal-preconditions)

(:types 
    physob
)

(:predicates
    (ontable ?x - physob)
    (clear ?x - physob)
    (on ?x ?y - physob)
)

(:action move
    :parameters (?x ?y - physob)
    :precondition (and (clear ?x) (clear ?y))
    :effect (and (not (clear ?y))
                (forall (?z - physob)
                    (when (on ?x ?z)
                        (and (not (on ?x ?z))
                            (clear ?z))
                    )
                )
                (on ?x ?y)
            )
)

(:action moveToTable
    :parameters (?x - physob)
    :precondition (and (clear ?x) (not (ontable ?x)))
    :effect (and (ontable ?x)
                (forall (?z - physob)
                    (when (on ?x ?z)
                        (and (not (on ?x ?z))
                            (clear ?z))
                    )
                )
            )
)

)