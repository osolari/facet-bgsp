% SAIM unified professional paper class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{saim-paper}[2026/08/15 Computer Modern Sans technical-document system, v1.1]
\LoadClassWithOptions{article}

% Page and typography
% The two-column paper retains the publication geometry. The single-column
% report uses a narrower measure for comfortable long-form reading.
\makeatletter
\if@twocolumn
  \RequirePackage[top=20mm,bottom=22mm,left=22mm,right=22mm,columnsep=7.5mm]{geometry}
\else
  \RequirePackage[top=20mm,bottom=22mm,left=31mm,right=31mm]{geometry}
\fi
\makeatother
\input{../shared/brand/typography.tex}
\RequirePackage[english]{babel}
\RequirePackage{ragged2e}
\RequirePackage{etoolbox}
\RequirePackage{enumitem}
\RequirePackage{placeins}
\RequirePackage{eso-pic}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}

\setstretch{1.06}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.56em plus 0.10em minus 0.08em}
\setlength{\emergencystretch}{1.5em}
\raggedbottom
\setlist[itemize]{leftmargin=1.35em,itemsep=1.5pt,topsep=3pt,parsep=0pt}
\setlist[enumerate]{leftmargin=1.55em,itemsep=1.5pt,topsep=3pt,parsep=0pt}
\renewcommand{\labelitemi}{\textcolor{Primary}{\raisebox{0.18ex}{\scriptsize$\bullet$}}}

% Graphics and tables
\RequirePackage{graphicx}
\RequirePackage{subcaption}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{array}
\RequirePackage{multirow}
\renewcommand{\arraystretch}{1.24}
\setlength{\tabcolsep}{5.2pt}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Color and shared components
\RequirePackage{xcolor}
\input{../shared/brand/colors.tex}
\input{../shared/components/frontmatter.tex}
\RequirePackage[most]{tcolorbox}
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds,shapes.geometric}
\input{../shared/components/boxes.tex}
\input{../shared/components/evidence.tex}
\color{Ink}

% References
\RequirePackage[round,authoryear]{natbib}
\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks=true,
  linkcolor=PrimaryDark,
  citecolor=PrimaryDark,
  urlcolor=Secondary,
  pdfborder={0 0 0}
}
\def\bibfont{\small}

% Section hierarchy
\RequirePackage{titlesec}
\titleformat{\section}[hang]
  {\Large\rmfamily\bfseries\color{Ink}}
  {\color{Primary}\thesection}{0.55em}{}
\titleformat{\subsection}[hang]
  {\large\rmfamily\bfseries\color{Ink}}
  {\color{Primary}\thesubsection}{0.5em}{}
\titleformat{\subsubsection}[hang]
  {\normalsize\rmfamily\bfseries\color{Ink}}
  {\color{Primary}\thesubsubsection}{0.45em}{}
\titleformat{\paragraph}[runin]
  {\normalsize\rmfamily\bfseries\color{Ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{15pt plus 2pt minus 2pt}{6pt}
\titlespacing*{\subsection}{0pt}{11pt plus 2pt minus 1pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{8pt plus 1pt minus 1pt}{3pt}
\titlespacing*{\paragraph}{0pt}{7pt}{0.5em}

% Captions
\RequirePackage{caption}
\captionsetup{
  singlelinecheck=false,
  justification=raggedright,
  font={small},
  labelfont={bf,color=PrimaryDark},
  textfont={color=Muted},
  labelsep=period,
  skip=5pt
}

% Running head
\newcommand{\SAIMPaperShortTitle}{SAIM Research}
\newcommand{\shorttitle}[1]{\gdef\SAIMPaperShortTitle{#1}}
\newcommand{\SAIMPaperVersion}{}
\newcommand{\paperversion}[1]{\gdef\SAIMPaperVersion{#1}}
\setlength{\headheight}{18pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\rmfamily\scriptsize\color{Muted}\SAIMPaperShortTitle}
\fancyhead[R]{\rmfamily\scriptsize\color{Muted}\SAIMPaperVersion}
\fancyfoot[R]{\rmfamily\scriptsize\color{Muted}\thepage\,/\,\pageref*{LastPage}}
\renewcommand{\headrulewidth}{0.35pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{Rule}\leaders\hrule height \headrulewidth\hfill}}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyfoot[R]{\rmfamily\scriptsize\color{Muted}\thepage\,/\,\pageref*{LastPage}}
  \renewcommand{\headrulewidth}{0pt}}

% Metadata collection
\makeatletter
\renewcommand\addtolist[5][]{%
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}
\makeatother

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\SAIMFrontMetaFont\bfseries #2\ifstrempty{#1}{}{$^{#1}$}}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{\enspace\textcolor{RuleDark}{\textbullet}\enspace}}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\small\SAIMFrontMetaFont\color{Muted}\ifstrempty{#1}{}{$^{#1}$}#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{\enspace\textcolor{RuleDark}{\textbullet}\enspace}}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\footnotesize\SAIMFrontMetaFont\color{Muted}\ifstrempty{#1}{}{$^{#1}$}#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{\enspace}}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\footnotesize\SAIMFrontMetaFont\color{Muted}\textbf{#1}\enspace #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\hspace{1.4em}}}

\newcommand{\abstractlist}{}
\renewcommand{\abstract}[1]{\gdef\abstractlist{#1}}
\newcommand{\SAIMPaperOpeningLabel}{ABSTRACT}
\newcommand{\openinglabel}[1]{\gdef\SAIMPaperOpeningLabel{#1}}
\newcommand{\prelude}[1]{\openinglabel{PRELUDE}\abstract{#1}}
\newcommand{\opener}[1]{\openinglabel{OPENER}\abstract{#1}}
\newcommand{\keywordslist}{}
\newcommand{\keywords}[1]{\gdef\keywordslist{#1}}
\newcommand{\paperlabellist}{SAIM RESEARCH}
\newcommand{\paperlabel}[1]{\gdef\paperlabellist{#1}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}
\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}
\renewcommand{\title}[1]{\gdef\titlelist{#1}}
\providecommand{\titlelist}{Untitled research paper}

% Paper callouts
\newtcolorbox{keyclaim}[1][Key claim]{saim/primary,title={#1}}
\newtcolorbox{scopebox}[1][Scope and interpretation]{saim/blue,title={#1}}
\newtcolorbox{resultbox}[1][Result]{saim/green,title={#1}}
\newtcolorbox{cautionbox}[1][Caution]{saim/amber,title={#1}}
\newtcolorbox{paperbox}[1][Research note]{saim/neutral,title={#1}}

% Apply the same editorial rail to theorem-like environments declared by users.
\newcommand{\SAIMPaperTheoremStyle}[1]{%
  \tcolorboxenvironment{#1}{
    blanker,
    breakable,
    before skip=8pt,
    after skip=7pt,
    left=8pt,
    right=2pt,
    top=4pt,
    bottom=4pt,
    borderline west={1.8pt}{0pt}{Primary},
  }}


% Compact report table of contents. Intended for technical reports and internal
% papers where the reader benefits from seeing the structure before Section 1.
\setcounter{tocdepth}{2}
\makeatletter
\newcommand{\SAIMPaperTableOfContents}{%
  \vspace{4pt}%
  \begin{tcolorbox}[
    blanker,
    breakable,
    left=10pt,
    right=2pt,
    top=2pt,
    bottom=2pt,
    before skip=0pt,
    after skip=10pt,
    borderline west={2.2pt}{0pt}{Primary}
  ]
    {\SAIMNavigationFont\fontsize{8}{9.6}\selectfont\color{Primary}\textls[165]{CONTENTS}\par}
    \vspace{4pt}
    {\small\setlength{\parskip}{0pt}\@starttoc{toc}}%
  \end{tcolorbox}%
}
\makeatother

% Appendix heading. Preserve the active body layout rather than forcing the
% single-column report back into two columns.
\newcommand{\SAIMPaperAppendixHeading}{%
  \vspace*{2pt}
  {\rmfamily\bfseries\fontsize{24}{28}\selectfont\color{Ink}Appendix\par}
  \vspace{4pt}{\color{Primary}\rule{\textwidth}{1.6pt}}\vspace{10pt}%
}
\makeatletter
\newcommand{\beginappendix}{%
  \clearpage
  \appendix
  \if@twocolumn
    \twocolumn[\SAIMPaperAppendixHeading]
  \else
    \noindent\SAIMPaperAppendixHeading
  \fi
}
\makeatother

% Editorial-rail title opening
\newcommand{\SAIMFirstPageAccent}{%
  \AddToShipoutPictureBG*{%
    \AtPageUpperLeft{\color{Primary}\rule[-3.2pt]{\paperwidth}{3.2pt}}}}

\makeatletter
\newcommand{\SAIMPaperTitleBlock}{%
  \if@twocolumn
    {\SAIMFrontRagged\SAIMFrontTitleFont\fontsize{25}{29}\selectfont\color{Ink}\titlelist\par}%
  \else
    {\SAIMFrontRagged\SAIMFrontTitleFont\fontsize{23.5}{27.5}\selectfont\color{Ink}\titlelist\par}%
  \fi
}
\makeatother

\newcommand{\mymaketitle}{%
  \SAIMFirstPageAccent
  \thispagestyle{plain}
  \begingroup
  \setlength{\parindent}{0pt}
  \setlength{\parskip}{0pt}
  \noindent
  \begin{minipage}[c]{0.72\textwidth}
    {\SAIMNavigationFont\fontsize{8}{9.6}\selectfont\color{Primary}\textls[165]{\paperlabellist}}
  \end{minipage}%
  \begin{minipage}[c]{0.28\textwidth}
    \raggedleft\includegraphics[width=1.75cm]{../shared/brand/logos/saim-logo-color.png}
  \end{minipage}

  \vspace{9pt}
  \SAIMPaperTitleBlock
  \vspace{9pt}
  {\small\authorlist\par}
  \vspace{3pt}
  {\affiliationlist\par}
  \ifdefempty{\contributionlist}{}{\vspace{2pt}{\contributionlist\par}}

  \vspace{10pt}
  \begin{tcolorbox}[
    blanker,
    breakable,
    left=12pt,
    right=2pt,
    top=1pt,
    bottom=1pt,
    before skip=0pt,
    after skip=0pt,
    borderline west={3pt}{0pt}{Primary}
  ]
    {\SAIMNavigationFont\fontsize{8}{9.6}\selectfont\color{Primary}\textls[165]{\SAIMPaperOpeningLabel}\par}
    \vspace{6pt}
    {\small\color{Ink}\setstretch{1.07}\abstractlist\par}
    \ifdefempty{\keywordslist}{}{%
      \vspace{8pt}
      {\footnotesize\color{Muted}\SAIMFrontMetaFont\textbf{Keywords}\enspace \keywordslist\par}}
  \end{tcolorbox}

  \ifdefempty{\metadatalist}{}{\vspace{7pt}{\hbadness=10000\metadatalist\par}}
  \vspace{9pt}
  {\color{Rule}\rule{\textwidth}{0.6pt}}
  \vspace{7pt}
  \SAIMPaperTableOfContents
  {\color{Rule}\rule{\textwidth}{0.6pt}}
  \vspace{5pt}
  \endgroup
  \FloatBarrier
}

\makeatletter
\if@twocolumn
  % The abstract and contents now form a deliberate full-width opening page.
  % Switch back to the publication body only after that page is complete.
  \renewcommand{\maketitle}{\onecolumn\mymaketitle\twocolumn}
\else
  \renewcommand{\maketitle}{\mymaketitle}
\fi
\makeatother
