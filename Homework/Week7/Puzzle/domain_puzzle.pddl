

(define (domain puzzle)


(:requirements :strips :typing :equality :conditional-effects :universal-preconditions)

(:types 
    num loc
)
(:constants B - num) ; B is Blank

(:predicates
    (Block ?x - num)
    (Pos ?x - loc)
    (At ?x - num ?y - loc)
    (next ?x - loc ?y - loc)
)


(:action slide
    :parameters (?x - loc ?y - loc)
    :precondition (and (at B ?y) (next ?x ?y))
    :effect (and (forall (?z - num) ; ?z is num at ?x
                            (when (and (Block ?z) (at ?z ?x))
                                (and (not (at ?z ?x)) 
                                    (at ?z ?y))))
                (not (at B ?y)) (at B ?x)
            )
)


)