(define (problem prob) (:domain puzzle)
(:objects 
    B0 B1 B2 B3 B4 B5 B6 B7 B8 - num
    P1 P2 P3 P4 P5 P6 P7 P8 P9 - loc
)

(:init
    ;todo: put the initial state's facts and numeric values here
    (Block B1) (Block B2) (Block B3) (Block B4) (Block B5)
    (Block B6) (Block B7) (Block B8)
    (Pos P1) (Pos P2) (Pos P3) (Pos P4) (Pos P5)
    (Pos P6) (Pos P7) (Pos P8) (Pos P9)
    (At B1 P1) (At B2 P2) (At B3 P3)
    (At B7 P4) (At B8 P5) (At B P6)
    (At B6 P7) (At B4 P8) (At B5 P9)
    (next P1 P2) (next P2 P3) (next P4 P5) (next P5 P6)
    (next P7 P8) (next P8 P9) (next P1 P4) (next P2 P5)
    (next P3 P6) (next P4 P7) (next P5 P8) (next P6 P9)
    (next P2 P1) (next P3 P2) (next P5 P4) (next P6 P5)
    (next P8 P7) (next P9 P8) (next P4 P1) (next P5 P2)
    (next P6 P3) (next P7 P4) (next P8 P5) (next P9 P6)
)

(:goal (and
    ;todo: put the goal condition here
    (At B1 P1) (At B2 P2) (At B3 P3)
    (At B4 P4) (At B5 P5) (At B6 P6)
    (At B7 P7) (At B8 P8) (At B P9)
    )
)

;un-comment the following line if metric is needed
;(:metric minimize (???))
)