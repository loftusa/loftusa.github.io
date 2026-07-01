import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";

const POSTS_DIR = path.join(process.cwd(), "content", "posts");

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeSlug)
  .use(rehypeAutolinkHeadings, { behavior: "wrap" })
  .use(rehypeKatex)
  .use(rehypeHighlight, { detect: true, ignoreMissing: true })
  .use(rehypeStringify, { allowDangerousHtml: true });

export async function renderMarkdown(md: string): Promise<string> {
  return String(await processor.process(md));
}

export type PostMeta = {
  slug: string[];
  permalink: string;
  title: string;
  date: string; // ISO
  excerpt: string;
};
export type Post = PostMeta & { html: string; mathjax: boolean; comments: boolean };

function permalinkToSlug(permalink: string): string[] {
  const parts = permalink.replace(/^\/+|\/+$/g, "").split("/").filter(Boolean);
  if (parts[0] === "posts") parts.shift();
  return parts;
}

function isoDate(d: unknown): string {
  if (!d) return "";
  const dt = new Date(d as string);
  return Number.isNaN(dt.getTime()) ? String(d) : dt.toISOString();
}

function deriveExcerpt(content: string, fallback?: unknown): string {
  if (typeof fallback === "string" && fallback.trim()) return fallback.trim();
  for (const block of content.split(/\n{2,}/)) {
    const t = block.trim();
    if (!t || t.startsWith("#") || t.startsWith("<") || t.startsWith("$$")) continue;
    const text = t
      .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/[*_`>#]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (text) return text.length > 180 ? text.slice(0, 180).trimEnd() + "…" : text;
  }
  return "";
}

function stripRedundantH1(content: string, title: string): string {
  const m = content.match(/^\s*#\s+([^\n]+)\n+/);
  if (m && m[1].trim().toLowerCase() === title.trim().toLowerCase()) {
    return content.slice(m[0].length);
  }
  return content;
}

function postFiles(): string[] {
  if (!fs.existsSync(POSTS_DIR)) return [];
  return fs.readdirSync(POSTS_DIR).filter((f) => f.endsWith(".md"));
}

function readMeta(file: string): PostMeta & { _content: string; _data: matter.GrayMatterFile<string>["data"] } {
  const raw = fs.readFileSync(path.join(POSTS_DIR, file), "utf8");
  const { data, content } = matter(raw);
  const permalink: string = data.permalink || `/posts/${file.replace(/\.md$/, "")}/`;
  const title: string = data.title || file.replace(/\.md$/, "");
  return {
    slug: permalinkToSlug(permalink),
    permalink,
    title,
    date: isoDate(data.date),
    excerpt: deriveExcerpt(content, data.excerpt),
    _content: content,
    _data: data,
  };
}

export function getAllPostsMeta(): PostMeta[] {
  return postFiles()
    .map((f) => {
      const m = readMeta(f);
      return { slug: m.slug, permalink: m.permalink, title: m.title, date: m.date, excerpt: m.excerpt };
    })
    .sort((a, b) => (a.date < b.date ? 1 : -1));
}

export async function getPostBySlug(slug: string[]): Promise<Post | null> {
  const target = "/posts/" + slug.join("/");
  for (const f of postFiles()) {
    const m = readMeta(f);
    if (m.permalink.replace(/\/+$/, "") === target.replace(/\/+$/, "")) {
      const body = stripRedundantH1(m._content, m.title);
      const html = await renderMarkdown(body);
      return {
        slug: m.slug,
        permalink: m.permalink,
        title: m.title,
        date: m.date,
        excerpt: m.excerpt,
        html,
        mathjax: !!m._data.mathjax,
        comments: m._data.comments !== false,
      };
    }
  }
  return null;
}

export function getAllPostSlugs(): string[][] {
  return postFiles().map((f) => readMeta(f).slug);
}
