import ArticleLayout from './components/ArticleLayout';
import Section from './components/Section';
import { FileEarmarkPdf, Github } from 'react-bootstrap-icons';

import Fig1 from './assets/figures/scaling_graph.png';

import { Prism as SyntaxHighligher } from 'react-syntax-highlighter';
import { a11yDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import InteractiveFigure1 from './interactive_figures/InteractiveFigure1';

function App() {
  return <ArticleLayout >{({ Citation, CitationBank }) => <>
    <h2 className='text-center'>Forecasting Frontier Language Model Agent Capabilities</h2>
    <section id="authors" className="d-flex justify-content-center">
      <div className='p-2 text-center'>
        <a className="fs-5" href="https://pimpale.github.io">Govind Pimpale</a><sup>*</sup>,
        <div className="fs-6">
          MATS
        </div>
      </div>
      <div className='p-2 text-center'>
        <a className="fs-5" href="https://www.linkedin.com/in/axelhojmark/">Axel Højmark</a><sup>*</sup>,
        <div className="fs-6">
          MATS & Apollo Research
        </div>
      </div>
      <div className='p-2 text-center'>
        <a className="fs-5" href="https://www.linkedin.com/in/j%C3%A9r%C3%A9my-scheurer-927563b0/">Jérémy Scheurer</a><sup>&dagger;</sup>,
        <div className="fs-6">
          Apollo Research
        </div>
      </div>
      <div className='p-2 text-center'>
        <a className="fs-5" href="https://www.linkedin.com/in/marius-hobbhahn-128927175/">Marius Hobbhahn</a><sup>&dagger;</sup>
        <div className="fs-6">
          Apollo Research
        </div>
      </div>
    </section>
    <section id="footnotes" className="text-center">
      <sup>*</sup> Equal contribution
      <sup className='ms-2'>&dagger;</sup> Equal supervision
    </section>
    <section id="links" className="d-flex justify-content-center">
      <div className='p-3'>
        <a className='btn btn-outline-secondary p-2 fs-5 d-flex align-items-center' href="https://arxiv.org/abs/">
          <FileEarmarkPdf className='me-1' /> ArXiv
        </a>
      </div>
      <div className='p-3'>
        <a className='btn btn-outline-secondary p-2 fs-5 d-flex align-items-center' href="https://github.com/hojmax/agent-scaling-laws">
          <Github className='me-1' /> Code
        </a>
      </div>
    </section>
    <section id="fig1" className='d-flex justify-content-center'>
      <img src={Fig1} className='w-75' />
    </section>
    <section id="tldr" className="mt-5 px-5">
      <b>TLDR:</b> We evaluate different methods for forecasting frontier language model capabilities, finding that by early 2026, advanced AI agents could achieve 87% on SWE-Bench Verified, M% on CyBench, and N on RE-Bench.
    </section>
    <Section id="abstract" name="Abstract">
      As language models (LMs) increasingly operate as autonomous agents, accurately forecasting their capabilities becomes crucial for societal preparedness.
      We evaluate six forecasting methods that predict downstream capabilities of LM agents.
      We use "one-step" approaches that predict benchmark scores from input metrics like compute or model release date directly or "two-step" approaches that first predict an intermediate metric like the principal component of cross-benchmark performance (PC-1) and human-evaluated competitive Elo ratings.
      We evaluate our forecasting methods by backtesting them on a dataset of 38 LMs from the OpenLLM 2 leaderboard.
      We then use the validated two-step approach (Release Date <ArrowRight /> Elo <ArrowRight /> Benchmark) to predict LM agent performance for frontier models on three benchmarks:
      SWEBench (software development), Cybench (cybersecurity assessment), and RE-Bench (ML research engineering).
      Our forecast predicts that by the beginning of 2026, non-specialized LM agents with low capability elicitation will reach a success rate of 54% on SWE-Bench Verified, while state-of-the-art LM agents will reach an 87% success rate.
      Our approach does not account for recent advances in inference-compute scaling and might thus be too conservative.
    </Section>
    <Section id="interactive-fig1" name="Explore the Forecast">
      <InteractiveFigure1 />
    </Section>
    <Section id="interactive-fig1" name="Conditional Probability Distributions">
      <AsideCard title={"Explanation"} >
        <p>
          Each of these graphs are a horizontal slice through Figure 1.
          The vertical lines mark the mainline forecast,
          and the translucent curves show the probability of the corresponding benchmark reaching a threshold score (0.9 for Cybench and SWE-Bench Verified, 1.0 for RE-Bench) at each date.
          You can adjust the benchmark threshold by using the slider at the bottom of each graph.
        </p>
      </AsideCard>
      <InteractiveHistogram />
    </Section>
    <Section id="limitations" name="Limitations">
      <p>
        <b>Paradigm Changes</b>: While this paper does not make any explicit assumptions about the training paradigm of any particular model,
        we fit almost all predictions on models that were trained with the "pre-training scaling" paradigm, where the
        primary driver for downstream performance was improvements of pre-training. However, with OpenAI's o1, we may
        start to see a new "inference scaling" paradigm where models are trained to utilize inference compute much more
        effectively through reasoning. This might invalidate our predictions and thus provide a reason to assume faster
        progress than our forecasts would suggest, even for high-elicitation predictions.
      </p>
      <p>
        <b>Underelicitation</b>:
        As discussed in the elicitation section, we did not put a lot of effort into elicitation. As a consequence,
        we know that our results are significantly below frontier performance and that our "low-elicitation" predictions
        are conservative. Even the "max-current-elicitation" forecast might underestimate performance due to paradigm
        changes or later breakthroughs in agent scaffolding and elicitation.
      </p>

      <p>
        <b>Small sample size</b>:
        Unfortunately, almost by definition, there are only a small number of frontier models. Therefore, our predictions
        have a small sample size. This is partially mitigated by making use of the two-step methodology and predicting
        the intermediate variable independently. However, we think the small sample size should imply large uncertainty
        about our forecasts.
        This limitation also affects our backtesting. Because our available test data is limited, we must rely on small
        evaluation windows, some as brief as two-month intervals. As a result, we have little empirical evidence regarding
        how our predictions might perform over longer periods.
      </p>
      <p>
        <b>Limited Scope of Evaluations</b>:
        The benchmarks we consider focus primarily on software engineering, cyber capabilities, and machine learning
        engineering. Noteworthy other agent benchmarks include GAIA<Citation source='https://arxiv.org/abs/2311.12983' /> and OS-World<Citation source='https://arxiv.org/abs/2404.07972' /> for browsing and tool use, as well as
        MLE-Bench<Citation source='https://openai.com/index/mle-bench/' /> for additional machine learning capabilities.
      </p>
    </Section>

    <Section id="bibtex" name="Bibtex">
      <SyntaxHighligher
        className="mx-5 my-5"
        language={"bibtex"}
        showLineNumbers
        style={a11yDark}
        children={dedent`
          @article{pimpale2026forecasting,
            title={Forecasting Frontier Language Model Agent Capabilities},
            author={Pimpale, Govind and Højmark, Axel and Scheurer, Jérémy and Hobbhahn, Marius},
            journal={arXiv preprint arXiv:},
            year={2026}
          }
        `}
      />
    </Section>
    <Section id="references" name="References">
      <CitationBank />
    </Section>
  </>
  }</ArticleLayout>
}




import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import './styles/style.scss';
import 'bootstrap/dist/js/bootstrap';
import { ArrowRight } from 'react-bootstrap-icons';
import dedent from 'dedent';
import InteractiveHistogram from './interactive_figures/InteractiveHistogram';
import AsideCard from './components/AsideCard';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
